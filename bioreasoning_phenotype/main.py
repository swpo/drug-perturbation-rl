"""load_environment entry point for the small-molecule chain env."""

from typing import Any

import verifiers as vf

from bioreasoning_phenotype.env import load_examples, make_rubric
from bioreasoning_phenotype.hallmarks import describe_hallmark
from bioreasoning_phenotype.strategy_judge import build_strategy_judge_runtime
from bioreasoning_phenotype.tools import identify_compound


def load_environment(
    entry_points: list[str] | None = None,
    phenotypes: list[str] | None = None,
    cell_lines: list[str] | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    reward_weights: dict[str, float] | None = None,
    tools: bool = False,
    hallmark_tools: bool = False,
    max_tool_turns: int = 5,
    strategy_judge_mode: str = "off",
    strategy_judge_model: str = "google/gemini-3-flash-preview",
    strategy_judge_base_url: str = "https://api.pinference.ai/api/v1",
    strategy_judge_api_key_var: str = "PRIME_API_KEY",
    strategy_judge_concurrency: int = 32,
    strategy_judge_max_tokens: int = 600,
    strategy_judge_max_retries: int = 2,
    strategy_judge_timeout_seconds: float = 180.0,
    **kwargs: Any,
) -> vf.Environment:
    """Build the small-molecule reasoning-chain env.

    Skinny chain: each example asks for target → MoA → pathways+direction → ONE
    phenotype prediction. The phenotype varies per example.

    Args:
        entry_points: subset of full-chain, ablation, and standalone curriculum
            tasks. Common values include {"smiles_only", "from_target",
            "from_moa", "from_pathways", "phenotype_direct",
            "target_from_smiles", "moa_from_target", "pathways_from_smiles",
            "pathways_from_moa", "phenotype_from_pathways"}. Default None = all
            available entry points.
        phenotypes: subset of {"viability", "cell_cycle", "stress", "magnitude"}.
            Default None = all four.
        cell_lines: subset of the 6 LINCS core lines
            {A375, A549, HEPG2, HT29, MCF7, PC3}. Default None = all.
        num_train_examples / num_eval_examples: -1 for all; else downsample.
        reward_weights: aggregate-reward weights for the chain steps. Keys:
            target / moa / pathways / phenotype. Default 0.15/0.15/0.25/0.45.
        tools: if True, expose `identify_compound(smiles)` so the model can
            look up compound identity and chemistry metadata during reasoning.
        hallmark_tools: if True, expose `describe_hallmark(pathway)` so the
            model can look up Hallmark ontology metadata during reasoning.
        max_tool_turns: max number of tool-call rounds when any tool is enabled.
        strategy_judge_mode: ``off``, ``shadow`` (measure only), or ``gate``
            (multiply deterministic reward by the proof-validated v0.27 gate).
        strategy_judge_*: locked Prime/Gemini runtime controls for the bounded
            on-policy strategy-judge experiment. Judge failures fail open.
        kwargs: absorbed for forward-compat with test harnesses.
    """
    train_ds, eval_ds = load_examples(
        entry_points=entry_points,
        phenotypes=phenotypes,
        cell_lines=cell_lines,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
    )
    if strategy_judge_mode != "off":
        vf.ensure_keys([strategy_judge_api_key_var])
    strategy_judge = build_strategy_judge_runtime(
        mode=strategy_judge_mode,
        model=strategy_judge_model,
        base_url=strategy_judge_base_url,
        api_key_var=strategy_judge_api_key_var,
        concurrency=strategy_judge_concurrency,
        max_tokens=strategy_judge_max_tokens,
        max_retries=strategy_judge_max_retries,
        timeout_seconds=strategy_judge_timeout_seconds,
    )
    rubric = make_rubric(
        weights=reward_weights,
        strategy_judge=strategy_judge,
        strategy_judge_mode=strategy_judge_mode,
    )

    tool_fns = []
    if tools:
        tool_fns.append(identify_compound)
    if hallmark_tools:
        tool_fns.append(describe_hallmark)

    if tool_fns:
        return vf.ToolEnv(
            dataset=train_ds,
            eval_dataset=eval_ds,
            rubric=rubric,
            tools=tool_fns,
            max_turns=max_tool_turns,
        )

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
    )
