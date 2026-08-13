"""ChainEnv — multi-step reasoning env on small-molecule perturbations (v0.10).

Skinny chain: each full-chain example asks for the same 3-step causal upstream
(target → MoA → pathways+direction) plus ONE downstream phenotype prediction.
Standalone curriculum examples can request any subset of those outputs.

Upstream grading (always, when not pre-filled by entry point):
  Step 1 (target):  F1 on gene-symbol set vs Drug Repurposing Hub
  Step 2 (MoA):     exact normalized string match vs DRH `moa` column
  Step 3 (pathway): F1 on (pathway_name, direction) tuples

Downstream grading, when requested (one of):
  viability:   piecewise-linear on |pred_LFC − GT_LFC|
  cell_cycle:  exact match on 3-class
  stress:      exact match on 4-class
  magnitude:   exact match on 3-class
"""
import json
import re
from pathlib import Path

import pandas as pd
import verifiers as vf
from datasets import Dataset

from bioreasoning_phenotype.hallmarks import hallmark_names
from bioreasoning_phenotype.process_judge import (
    PROCESS_JUDGE_MODES,
    ProcessJudgeRuntime,
)
from bioreasoning_phenotype.prompts import (
    CYCLE_TAG, MAGNITUDE_TAG, MOA_TAG, PATHWAYS_TAG, STRESS_TAG,
    TARGET_TAG, VIABILITY_TAG,
    CYCLE_CLASSES, MAGNITUDE_CLASSES, STRESS_CLASSES,
    SYSTEM_PROMPT,
)
from bioreasoning_phenotype.strategy_judge import (
    STRATEGY_JUDGE_MODES,
    StrategyJudgeRuntime,
)

DATA_PATH = Path(__file__).parent / "data" / "smallmol_chain_examples.parquet"


# Per-step reward weights for the aggregate (must sum to 1.0).
# Upstream weights apply only when the step is requested (not pre-filled).
# Downstream weight always applies (one downstream step per example).
DEFAULT_REWARD_WEIGHTS = {
    "target":    0.15,
    "moa":       0.15,
    "pathways":  0.25,
    "phenotype": 0.45,
}

# Viability scoring: piecewise-linear, full credit within ±0.25 LFC,
# zero credit beyond ±2.0.
VIABILITY_TOL_FULL = 0.25
VIABILITY_TOL_ZERO = 2.0


def _last_assistant(completion) -> str:
    """Get the last assistant message's content.

    Verifiers passes Pydantic AssistantMessage objects at scoring time; local
    JSON-loaded fixtures pass plain dicts. Both support .get(), so duck-type.
    """
    if isinstance(completion, list):
        for msg in reversed(completion):
            if hasattr(msg, "get") and msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    return content
        return ""
    return str(completion or "")


def _extract_tag(text: str, tag: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _parse_list(text: str | None) -> list[str]:
    """Comma- or pipe-separated tokens, uppercased."""
    if not text:
        return []
    parts = re.split(r"[|,]", text)
    return [p.strip().upper() for p in parts if p.strip()]


def _parse_pathway_pairs(text: str | None) -> list[tuple[str, str]]:
    """Parse 'HALLMARK_X:up, HALLMARK_Y:down' into [(name, dir), ...]."""
    if not text:
        return []
    out = []
    for token in re.split(r"[,;\n]", text):
        token = token.strip()
        if not token:
            continue
        # split on the final ':' to separate name from direction
        if ":" in token:
            name, direction = token.rsplit(":", 1)
            name = name.strip().upper()
            direction = direction.strip().lower()
            if direction in ("up", "down") and name:
                out.append((name, direction))
    return out


def _exact_hallmark_name(name: str | None) -> str | None:
    """Return the canonical Hallmark name for exact names only, no aliases."""
    if not name:
        return None
    key = re.sub(r"[^A-Z0-9]+", "_", name.strip().upper())
    key = re.sub(r"_+", "_", key).strip("_")
    names = set(hallmark_names())
    if key in names:
        return key
    prefixed = f"HALLMARK_{key}"
    if not key.startswith("HALLMARK_") and prefixed in names:
        return prefixed
    return None


def _pathway_name_token(name: str) -> str:
    """Canonical token for valid names; unique invalid token otherwise."""
    canonical = _exact_hallmark_name(name)
    return canonical if canonical is not None else f"INVALID::{name.strip().upper()}"


def _f1_set(pred: set, gt: set) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    return 2 * prec * rec / (prec + rec)


def _normalize_moa(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.lower().strip().split())


def _score_target(pred_text: str | None, gt: list[str]) -> float:
    if not gt:
        return 0.0
    if pred_text is None:
        return 0.0
    return _f1_set(set(_parse_list(pred_text)), set(g.upper() for g in gt))


def _score_moa(pred_text: str | None, gt: str | None) -> float:
    if not gt:
        return 0.0
    if pred_text is None:
        return 0.0
    return 1.0 if _normalize_moa(pred_text) == _normalize_moa(gt) else 0.0


def _score_pathways(pred_text: str | None, gt_pairs: list[tuple[str, str]]) -> float:
    if not gt_pairs:
        return 0.0
    if pred_text is None:
        return 0.0
    pred = set(_parse_pathway_pairs(pred_text))
    gt = set((name.upper().replace("HALLMARK_", ""), direction)
             for name, direction in gt_pairs)
    pred = set((name.replace("HALLMARK_", ""), direction) for name, direction in pred)
    return _f1_set(pred, gt)


def _score_pathway_name_validity(pred_text: str | None) -> float:
    """Fraction of parsed pathway predictions that are exact Hallmark names."""
    pred = _parse_pathway_pairs(pred_text)
    if not pred:
        return 0.0
    valid = sum(1 for name, _ in pred if _exact_hallmark_name(name) is not None)
    return valid / len(pred)


def _score_pathway_name_f1(pred_text: str | None, gt_pairs: list[tuple[str, str]]) -> float:
    """F1 on Hallmark names only, ignoring direction but penalizing invalid names."""
    if not gt_pairs:
        return 0.0
    pred_names = {_pathway_name_token(name) for name, _ in _parse_pathway_pairs(pred_text)}
    gt_names = {_exact_hallmark_name(name) or name.upper() for name, _ in gt_pairs}
    return _f1_set(pred_names, gt_names)


def _score_pathway_direction_accuracy(
    pred_text: str | None,
    gt_pairs: list[tuple[str, str]],
) -> float:
    """Direction accuracy among exact Hallmark names shared by pred and GT."""
    if not gt_pairs:
        return 0.0
    pred_dirs: dict[str, set[str]] = {}
    for name, direction in _parse_pathway_pairs(pred_text):
        canonical = _exact_hallmark_name(name)
        if canonical is not None:
            pred_dirs.setdefault(canonical, set()).add(direction)

    gt_dirs = {
        (_exact_hallmark_name(name) or name.upper()): direction
        for name, direction in gt_pairs
    }
    overlap = set(pred_dirs) & set(gt_dirs)
    if not overlap:
        return 0.0
    correct = sum(1 for name in overlap if gt_dirs[name] in pred_dirs[name])
    return correct / len(overlap)


def _score_viability(pred_text: str | None, gt_lfc: float | None) -> float:
    if gt_lfc is None:
        return 0.0
    if pred_text is None:
        return 0.0
    # Pull out the first signed-float token
    m = re.search(r"-?\d+\.?\d*", pred_text)
    if not m:
        return 0.0
    try:
        pred = float(m.group(0))
    except ValueError:
        return 0.0
    err = abs(pred - gt_lfc)
    if err <= VIABILITY_TOL_FULL:
        return 1.0
    if err >= VIABILITY_TOL_ZERO:
        return 0.0
    # piecewise-linear interpolation in between
    return 1.0 - (err - VIABILITY_TOL_FULL) / (VIABILITY_TOL_ZERO - VIABILITY_TOL_FULL)


def _score_class(pred_text: str | None, gt: str | None, classes: list[str]) -> float:
    """Exact match on a closed class set, case-insensitive."""
    if not gt:
        return 0.0
    if pred_text is None:
        return 0.0
    pred = pred_text.strip().lower()
    gt_norm = gt.strip().lower()
    classes_norm = [c.lower() for c in classes]
    if pred not in classes_norm:
        return 0.0
    return 1.0 if pred == gt_norm else 0.0


# Phenotype dispatch
PHENOTYPE_TAGS = {
    "viability":  VIABILITY_TAG,
    "cell_cycle": CYCLE_TAG,
    "stress":     STRESS_TAG,
    "magnitude":  MAGNITUDE_TAG,
}
PHENOTYPE_CLASSES = {
    "cell_cycle": CYCLE_CLASSES,
    "stress":     STRESS_CLASSES,
    "magnitude":  MAGNITUDE_CLASSES,
}


def _score_phenotype(phenotype: str, pred_text: str | None, gt) -> float:
    if phenotype == "viability":
        return _score_viability(pred_text, gt)
    return _score_class(pred_text, gt, PHENOTYPE_CLASSES[phenotype])


def _gt_from_answer(answer):
    return json.loads(answer) if isinstance(answer, str) else answer


def make_rubric(
    weights: dict[str, float] | None = None,
    strategy_judge: StrategyJudgeRuntime | None = None,
    strategy_judge_mode: str = "off",
    process_judge: ProcessJudgeRuntime | None = None,
    process_judge_mode: str = "off",
) -> vf.Rubric:
    """Build the chain rubric: aggregate reward + per-step metrics."""
    if strategy_judge_mode not in STRATEGY_JUDGE_MODES:
        raise ValueError(
            f"strategy_judge_mode must be one of {sorted(STRATEGY_JUDGE_MODES)}"
        )
    if strategy_judge_mode != "off" and strategy_judge is None:
        raise ValueError("shadow/gate strategy_judge_mode requires a judge runtime")
    if process_judge_mode not in PROCESS_JUDGE_MODES:
        raise ValueError(
            f"process_judge_mode must be one of {sorted(PROCESS_JUDGE_MODES)}"
        )
    if process_judge_mode != "off" and process_judge is None:
        raise ValueError("multiply process_judge_mode requires a judge runtime")
    if strategy_judge_mode != "off" and process_judge_mode != "off":
        raise ValueError("strategy_judge_mode and process_judge_mode are mutually exclusive")
    w = weights or DEFAULT_REWARD_WEIGHTS

    async def aggregate_reward(completion, answer, state, **_) -> float:
        """Weighted sum across requested chain/curriculum steps."""
        text = _last_assistant(completion)
        gt = _gt_from_answer(answer)

        total = 0.0
        total_w = 0.0
        # Upstream — only count when GT carries that step (i.e. the model was asked to predict it)
        if gt.get("target"):
            total += w["target"] * _score_target(_extract_tag(text, TARGET_TAG), gt["target"])
            total_w += w["target"]
        if gt.get("moa"):
            total += w["moa"] * _score_moa(_extract_tag(text, MOA_TAG), gt["moa"])
            total_w += w["moa"]
        if gt.get("pathways_signed"):
            pairs = [tuple(p) for p in gt["pathways_signed"]]
            total += w["pathways"] * _score_pathways(_extract_tag(text, PATHWAYS_TAG), pairs)
            total_w += w["pathways"]
        phenotype = gt.get("phenotype")
        if phenotype:
            phen_tag = PHENOTYPE_TAGS[phenotype]
            phen_gt = gt.get(_phenotype_gt_key(phenotype))
            total += w["phenotype"] * _score_phenotype(phenotype, _extract_tag(text, phen_tag), phen_gt)
            total_w += w["phenotype"]
        score = total / total_w if total_w > 0 else 0.0
        state["deterministic_reward"] = score
        return score

    async def strategy_judge_operational_gate(prompt, completion, state, **_) -> float:
        if strategy_judge is None:
            state["strategy_judge_operational_gate"] = 1.0
            return 1.0
        return await strategy_judge.evaluate(
            prompt=prompt,
            completion=completion,
            state=state,
        )

    async def process_judge_score(prompt, completion, info, state, **_) -> float:
        if process_judge is None:
            state["process_judge_score"] = 1.0
            return 1.0
        entry_point = str((info or {}).get("entry_point") or "")
        return await process_judge.evaluate(
            prompt=prompt,
            completion=completion,
            entry_point=entry_point,
            state=state,
        )

    async def training_reward(state, **_) -> float:
        deterministic = float(state.get("deterministic_reward", 0.0))
        if strategy_judge_mode == "gate":
            return deterministic * float(
                state.get("strategy_judge_operational_gate", 1.0)
            )
        if process_judge_mode == "multiply":
            return deterministic * float(state.get("process_judge_score", 1.0))
        return deterministic

    def _state_metric(name: str):
        async def metric(state, **_) -> float:
            return float(state.get(name, 0.0))

        metric.__name__ = name
        return metric

    async def target_f1(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("target"):
            return 0.0
        return _score_target(_extract_tag(_last_assistant(completion), TARGET_TAG), gt["target"])

    async def moa_accuracy(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("moa"):
            return 0.0
        return _score_moa(_extract_tag(_last_assistant(completion), MOA_TAG), gt["moa"])

    async def pathway_signed_f1(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("pathways_signed"):
            return 0.0
        pairs = [tuple(p) for p in gt["pathways_signed"]]
        return _score_pathways(_extract_tag(_last_assistant(completion), PATHWAYS_TAG), pairs)

    async def pathway_name_validity(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("pathways_signed"):
            return 0.0
        return _score_pathway_name_validity(_extract_tag(_last_assistant(completion), PATHWAYS_TAG))

    async def pathway_name_f1(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("pathways_signed"):
            return 0.0
        pairs = [tuple(p) for p in gt["pathways_signed"]]
        return _score_pathway_name_f1(_extract_tag(_last_assistant(completion), PATHWAYS_TAG), pairs)

    async def pathway_direction_accuracy(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        if not gt.get("pathways_signed"):
            return 0.0
        pairs = [tuple(p) for p in gt["pathways_signed"]]
        return _score_pathway_direction_accuracy(
            _extract_tag(_last_assistant(completion), PATHWAYS_TAG), pairs
        )

    async def phenotype_score(completion, answer, **_) -> float:
        gt = _gt_from_answer(answer)
        phenotype = gt.get("phenotype")
        if not phenotype:
            return 0.0
        phen_tag = PHENOTYPE_TAGS[phenotype]
        phen_gt = gt.get(_phenotype_gt_key(phenotype))
        return _score_phenotype(phenotype, _extract_tag(_last_assistant(completion), phen_tag), phen_gt)

    async def format_compliance(completion, answer, **_) -> float:
        """Fraction of requested answer tags present in the output."""
        text = _last_assistant(completion)
        gt = _gt_from_answer(answer)
        checks = []
        if gt.get("target"):
            checks.append(_extract_tag(text, TARGET_TAG) is not None)
        if gt.get("moa"):
            checks.append(_extract_tag(text, MOA_TAG) is not None)
        if gt.get("pathways_signed"):
            checks.append(_extract_tag(text, PATHWAYS_TAG) is not None)
        if gt.get("phenotype"):
            checks.append(_extract_tag(text, PHENOTYPE_TAGS[gt["phenotype"]]) is not None)
        return float(sum(checks)) / len(checks) if checks else 0.0

    rubric = vf.Rubric()
    # Ordering is intentional: deterministic score -> one judge call -> final
    # training reward. Verifiers executes functions in this order while sharing
    # the mutable rollout state.
    rubric.add_metric(aggregate_reward)
    if strategy_judge is not None:
        rubric.add_metric(strategy_judge_operational_gate)
        for metric_name in (
            "strategy_judge_parse_success",
            "strategy_judge_raw_invalid",
            "strategy_judge_proof_accepted_invalid",
            "strategy_judge_proof_rejected_invalid",
            "strategy_judge_failure",
            "strategy_judge_latency_seconds",
            "strategy_judge_input_tokens",
            "strategy_judge_output_tokens",
            "strategy_judge_violation_shotgun_enumeration",
            "strategy_judge_violation_tool_contradiction",
            "strategy_judge_violation_input_substitution",
            "strategy_judge_violation_answer_trace_contradiction",
            "strategy_judge_violation_unsupported_final_guess",
            "strategy_judge_violation_chain_endpoint_contradiction",
            "strategy_judge_violation_nonresponsive_or_malformed",
        ):
            rubric.add_metric(_state_metric(metric_name))
    if process_judge is not None:
        rubric.add_metric(process_judge_score)
        for metric_name in (
            "process_judge_node_score",
            "process_judge_transition_score",
            "process_judge_parse_success",
            "process_judge_reference_valid",
            "process_judge_fail_open",
            "process_judge_failure",
            "process_judge_support_adjustments",
            "process_judge_latency_seconds",
            "process_judge_input_tokens",
            "process_judge_output_tokens",
            "process_judge_node_supported_fraction",
            "process_judge_node_weak_fraction",
            "process_judge_node_unsupported_fraction",
            "process_judge_node_contradicted_fraction",
            "process_judge_transition_coherent_fraction",
            "process_judge_transition_weak_fraction",
            "process_judge_transition_disconnected_fraction",
            "process_judge_transition_contradicted_fraction",
        ):
            rubric.add_metric(_state_metric(metric_name))
    rubric.add_reward_func(training_reward, weight=1.0)
    rubric.add_metric(target_f1)
    rubric.add_metric(moa_accuracy)
    rubric.add_metric(pathway_signed_f1)
    rubric.add_metric(pathway_name_validity)
    rubric.add_metric(pathway_name_f1)
    rubric.add_metric(pathway_direction_accuracy)
    rubric.add_metric(phenotype_score)
    rubric.add_metric(format_compliance)
    return rubric


def _phenotype_gt_key(phenotype: str) -> str:
    """Which key in answer JSON holds the GT for this phenotype."""
    return {
        "viability":  "viability_lfc",
        "cell_cycle": "cell_cycle",
        "stress":     "stress",
        "magnitude":  "magnitude",
    }[phenotype]


def _build_dataset(df: pd.DataFrame) -> Dataset:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["user_prompt"]},
            ],
            "answer": r["answer_json"],
            "info": {
                "compound": r["compound"],
                "cell_line": r["cell_line"],
                "entry_point": r.get("entry_point", "smiles_only"),
                "phenotype": r.get("phenotype", "viability"),
            },
        })
    return Dataset.from_list(rows)


def load_eval_manifest(path: str | Path) -> Dataset:
    """Load an analysis-only evaluation dataset from a JSONL manifest.

    Each line must contain ``prompt`` (a Verifiers message list), ``answer``,
    and optional ``info``.  This leaves the packaged train/eval datasets and
    all default hosted behavior unchanged while allowing exact prompts retained
    from an earlier run to be rescored by a fresh base-model probe.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")

    rows = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        answer = row.get("answer")
        if not isinstance(prompt, list) or answer is None:
            raise ValueError(
                f"{manifest_path}:{line_number} requires prompt:list and answer"
            )
        info = row.get("info") or {}
        if not isinstance(info, dict):
            raise ValueError(f"{manifest_path}:{line_number} info must be an object")
        info = {**info, "prompt_key": row.get("prompt_key")}
        rows.append({"prompt": prompt, "answer": answer, "info": info})

    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {manifest_path}")
    return Dataset.from_list(rows)


def load_examples(
    entry_points: list[str] | None = None,
    phenotypes: list[str] | None = None,
    cell_lines: list[str] | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
) -> tuple[Dataset, Dataset]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/build_examples.py first."
        )
    df = pd.read_parquet(DATA_PATH)

    if entry_points is not None:
        df = df[df["entry_point"].isin(entry_points)]
    if phenotypes is not None:
        df = df[df["phenotype"].isin(phenotypes)]
    if cell_lines is not None:
        df = df[df["cell_line"].isin(cell_lines)]

    # Shuffle by default. Without this, vf-eval's "take first N" sampling lands on
    # only ~5 unique compounds (the parquet is sorted alphabetically by BRD ID, then
    # grouped by cell × phenotype, so first N hits the same first few compounds).
    # Random state 42 for reproducibility.
    train_df = df[df["split"] == "train"].sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df = df[df["split"] == "test"].sample(frac=1, random_state=42).reset_index(drop=True)

    if num_train_examples > 0 and len(train_df) > num_train_examples:
        train_df = train_df.sample(n=num_train_examples, random_state=42).reset_index(drop=True)
    if num_eval_examples > 0 and len(eval_df) > num_eval_examples:
        eval_df = eval_df.sample(n=num_eval_examples, random_state=42).reset_index(drop=True)

    return _build_dataset(train_df), _build_dataset(eval_df)
