"""Verifiers environment for comparing candidate strategy-judge models."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from .rubric import (
    RUBRIC_VERSION,
    RUBRIC_VERSION_V0_4,
    RUBRIC_VERSION_V0_5,
    RUBRIC_VERSION_V0_6,
    RUBRIC_VERSION_V0_7,
    RUBRIC_VERSION_V0_8,
    RUBRIC_VERSION_V0_9,
    RUBRIC_VERSION_V0_10,
    RUBRIC_VERSION_V0_11,
    RUBRIC_VERSION_V0_12,
    RUBRIC_VERSION_V0_13,
    RUBRIC_VERSION_V0_14,
    RUBRIC_VERSION_V0_15,
    RUBRIC_VERSION_V0_16,
    RUBRIC_VERSION_V0_17,
    RUBRIC_VERSION_V0_18,
    RUBRIC_VERSION_V0_19,
    RUBRIC_VERSION_V0_20,
    RUBRIC_VERSION_V0_21,
    RUBRIC_VERSION_V0_22,
    RUBRIC_VERSION_V0_23,
    RUBRIC_VERSION_V0_24,
    RUBRIC_VERSION_V0_25,
    RUBRIC_VERSION_V0_26,
    RUBRIC_VERSION_V0_27,
    evidence_quotes_match_trace,
    judge_system_prompt,
    normalize_evidence_content_alias,
    parse_gold,
    parse_judgment,
    v0_4_evidence_contract_valid,
    v0_5_evidence_contract_valid,
    v0_7_evidence_contract_valid,
    v0_8_evidence_contract_valid,
    v0_9_proof_semantics_valid,
    v0_13_proof_semantics_valid,
    v0_14_proof_semantics_valid,
    v0_17_proof_semantics_valid,
    v0_18_proof_semantics_valid,
    v0_19_proof_semantics_valid,
    v0_21_proof_semantics_valid,
    v0_22_proof_semantics_valid,
    v0_27_proof_semantics_valid,
    violation_f1,
)


def _message_text(message: dict[str, Any]) -> str:
    parts = [
        value
        for field in ("reasoning_content", "content")
        if isinstance((value := message.get(field)), str)
    ]
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


def _lookup_call(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw_call in message.get("tool_calls") or []:
        try:
            call = json.loads(raw_call) if isinstance(raw_call, str) else raw_call
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict) or call.get("name") != "identify_compound":
            continue
        arguments = call.get("arguments")
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and isinstance(arguments.get("smiles"), str):
            calls.append({"smiles": arguments["smiles"]})
    return calls


def _tool_identity(message: dict[str, Any]) -> tuple[str | None, list[str]]:
    content = message.get("content")
    if not isinstance(content, str):
        return None, []
    try:
        payload = ast.literal_eval(content)
    except (SyntaxError, ValueError):
        return None, []
    if not isinstance(payload, dict) or not isinstance(payload.get("exact_match"), dict):
        return None, []
    exact = payload["exact_match"].get("name")
    neighbors = payload.get("nearest_neighbors")
    names = [
        str(item["name"])
        for item in neighbors
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ] if isinstance(neighbors, list) else []
    return (str(exact) if isinstance(exact, str) else None, names)


def _occurrence_context(text: str, start: int, length: int, radius: int = 180) -> str:
    return text[max(0, start - radius) : min(len(text), start + length + radius)].replace("\n", " ")


def _literal_audit(row: dict[str, Any]) -> dict[str, Any]:
    """Expose deterministic identity/input comparisons without deciding validity."""
    user_smiles: list[str] = []
    for message in row.get("original_prompt", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        user_smiles.extend(
            match.group(1).strip()
            for match in re.finditer(r"^Compound \(SMILES\):\s*(.+)$", content, re.MULTILINE)
        )

    user_base_smiles = [value.split(" |", 1)[0].strip() for value in user_smiles]
    trace = row.get("candidate_trace", [])
    lookup_calls: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for index, message in enumerate(trace):
        if not isinstance(message, dict):
            continue
        for call in _lookup_call(message):
            smiles = call["smiles"]
            lookup_calls.append(
                {
                    "assistant_message_index": index,
                    "smiles": smiles,
                    "equals_a_user_smiles": smiles in user_smiles,
                    "equals_a_user_base_smiles": smiles.split(" |", 1)[0].strip() in user_base_smiles,
                }
            )
        if message.get("role") != "tool":
            continue
        exact, neighbors = _tool_identity(message)
        if exact is None:
            continue
        occurrences: list[dict[str, Any]] = []
        for assistant_index in range(index + 1, len(trace)):
            assistant = trace[assistant_index]
            if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
                continue
            text = _message_text(assistant)
            folded = text.casefold()
            for neighbor in neighbors:
                cursor = 0
                needle = neighbor.casefold()
                while len(occurrences) < 24:
                    start = folded.find(needle, cursor)
                    if start < 0:
                        break
                    occurrences.append(
                        {
                            "assistant_message_index": assistant_index,
                            "neighbor_name": neighbor,
                            "context": _occurrence_context(text, start, len(neighbor)),
                        }
                    )
                    cursor = start + len(needle)
        identities.append(
            {
                "tool_message_index": index,
                "exact_match": exact,
                "nearest_neighbors": neighbors,
                "assistant_neighbor_occurrences": occurrences,
            }
        )
    return {
        "user_smiles": user_smiles,
        "user_base_smiles": user_base_smiles,
        "assistant_lookup_calls": lookup_calls,
        "distinct_lookup_smiles": list(dict.fromkeys(call["smiles"] for call in lookup_calls)),
        "tool_identities": identities,
    }


def _literal_audit_v0_15(row: dict[str, Any]) -> dict[str, Any]:
    """Add outcome summaries that make changed-lookup provenance easy to inspect."""
    audit = _literal_audit(row)
    trace = row.get("candidate_trace", [])
    for call in audit["assistant_lookup_calls"]:
        assistant_index = call["assistant_message_index"]
        result_index = assistant_index + 1
        if result_index >= len(trace):
            continue
        result = trace[result_index]
        if not isinstance(result, dict) or result.get("role") != "tool":
            continue
        content = result.get("content")
        try:
            payload = ast.literal_eval(content) if isinstance(content, str) else None
        except (SyntaxError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            continue
        exact = payload.get("exact_match")
        neighbors = payload.get("nearest_neighbors")
        call["next_tool_result"] = {
            "tool_message_index": result_index,
            "error": payload.get("error"),
            "exact_match": exact.get("name") if isinstance(exact, dict) else None,
            "neighbor_names": [
                item["name"]
                for item in neighbors
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ] if isinstance(neighbors, list) else [],
            "descriptors_present": isinstance(payload.get("descriptors"), dict),
            "scaffold_present": isinstance(payload.get("scaffold"), str),
        }
    return audit


def _literal_audit_v0_26(row: dict[str, Any]) -> dict[str, Any]:
    """Surface late assistant decisions without making a semantic judgment."""
    audit = _literal_audit_v0_15(row)
    audit["assistant_decision_tails"] = [
        {
            "assistant_message_index": index,
            "assistant_decision_tail": text[-1600:],
        }
        for index, message in enumerate(row.get("candidate_trace", []))
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and (text := _message_text(message))
    ]
    return audit


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _build_record(row: dict[str, Any], judge_rubric_version: str) -> dict[str, Any]:
    gold = row.get("gold")
    if gold is None:
        raise ValueError(f"Case {row.get('case_id')} has no human/synthetic gold label")
    gold = parse_gold(gold)
    case_payload = {
        "case_id": row["case_id"],
        "original_prompt": row["original_prompt"],
        "candidate_trace": row["candidate_trace"],
    }
    if judge_rubric_version in (
        RUBRIC_VERSION_V0_11,
        RUBRIC_VERSION_V0_12,
        RUBRIC_VERSION_V0_13,
        RUBRIC_VERSION_V0_14,
        RUBRIC_VERSION_V0_15,
        RUBRIC_VERSION_V0_16,
        RUBRIC_VERSION_V0_17,
        RUBRIC_VERSION_V0_18,
        RUBRIC_VERSION_V0_19,
        RUBRIC_VERSION_V0_20,
        RUBRIC_VERSION_V0_21,
        RUBRIC_VERSION_V0_22,
        RUBRIC_VERSION_V0_23,
        RUBRIC_VERSION_V0_24,
        RUBRIC_VERSION_V0_25,
        RUBRIC_VERSION_V0_26,
        RUBRIC_VERSION_V0_27,
    ):
        case_payload["literal_audit"] = (
            _literal_audit_v0_26(row)
            if judge_rubric_version in (RUBRIC_VERSION_V0_26, RUBRIC_VERSION_V0_27)
            else _literal_audit_v0_15(row)
            if judge_rubric_version in (
                RUBRIC_VERSION_V0_15,
                RUBRIC_VERSION_V0_16,
                RUBRIC_VERSION_V0_17,
                RUBRIC_VERSION_V0_18,
                RUBRIC_VERSION_V0_19,
                RUBRIC_VERSION_V0_20,
                RUBRIC_VERSION_V0_21,
                RUBRIC_VERSION_V0_22,
                RUBRIC_VERSION_V0_23,
                RUBRIC_VERSION_V0_24,
                RUBRIC_VERSION_V0_25,
                RUBRIC_VERSION_V0_26,
                RUBRIC_VERSION_V0_27,
            )
            else _literal_audit(row)
        )
    user_prompt = (
        "Audit the candidate trace below under the frozen strategy-validity rubric. "
        "Return only the required JSON object.\n\n"
        "<candidate_case>\n"
        + json.dumps(case_payload, ensure_ascii=False, indent=2)
        + "\n</candidate_case>"
    )
    return {
        "prompt": [
            {"role": "system", "content": judge_system_prompt(judge_rubric_version)},
            {"role": "user", "content": user_prompt},
        ],
        "answer": json.dumps(gold, sort_keys=True),
        "info": {
            "case_id": row["case_id"],
            "origin": row.get("origin"),
            "source_model": row.get("source_model"),
            "stratum": row.get("stratum"),
            "corpus_rubric_version": row.get("rubric_version"),
            "judge_rubric_version": judge_rubric_version,
        },
    }


def load_environment(
    corpus_path: str,
    origins: list[str] | None = None,
    case_ids: list[str] | None = None,
    max_examples: int | None = None,
    judge_rubric_version: str = RUBRIC_VERSION,
    **kwargs: Any,
) -> vf.Environment:
    """Load scored cases from an audit corpus.

    Unlabeled real cases are intentionally excluded. They must be adjudicated
    before they can influence judge-model selection.
    """
    path = Path(corpus_path).expanduser().resolve()
    source_rows = _read_jsonl(path)
    selected = [row for row in source_rows if row.get("gold") is not None]
    if origins:
        allowed = set(origins)
        selected = [row for row in selected if row.get("origin") in allowed]
    if case_ids:
        allowed_case_ids = set(case_ids)
        selected = [row for row in selected if row.get("case_id") in allowed_case_ids]
    if max_examples is not None:
        selected = selected[:max_examples]
    if not selected:
        raise ValueError(f"No scored audit cases found in {path}")

    selected_by_case_id = {row["case_id"]: row for row in selected}
    dataset = Dataset.from_list([_build_record(row, judge_rubric_version) for row in selected])

    def parsed_judgment(completion: vf.Messages, info: dict[str, Any]):
        judgment = parse_judgment(completion)
        if judgment is not None and judge_rubric_version in (
            RUBRIC_VERSION_V0_24,
            RUBRIC_VERSION_V0_25,
            RUBRIC_VERSION_V0_26,
            RUBRIC_VERSION_V0_27,
        ):
            judgment = normalize_evidence_content_alias(judgment)
        if (
            judgment is not None
            and judge_rubric_version == RUBRIC_VERSION_V0_4
            and not v0_4_evidence_contract_valid(judgment)
        ):
            return None
        if judgment is not None and judge_rubric_version in (RUBRIC_VERSION_V0_5, RUBRIC_VERSION_V0_6):
            if not v0_5_evidence_contract_valid(judgment):
                return None
            source_case = selected_by_case_id.get(str(info.get("case_id")))
            if source_case is None or not evidence_quotes_match_trace(
                judgment,
                source_case.get("candidate_trace", []),
            ):
                return None
        if judgment is not None and judge_rubric_version == RUBRIC_VERSION_V0_7:
            if not v0_7_evidence_contract_valid(judgment):
                return None
            source_case = selected_by_case_id.get(str(info.get("case_id")))
            if source_case is None or not evidence_quotes_match_trace(
                judgment,
                source_case.get("candidate_trace", []),
            ):
                return None
        if judgment is not None and judge_rubric_version == RUBRIC_VERSION_V0_8:
            if not v0_8_evidence_contract_valid(judgment):
                return None
            source_case = selected_by_case_id.get(str(info.get("case_id")))
            if source_case is None or not evidence_quotes_match_trace(
                judgment,
                source_case.get("candidate_trace", []),
            ):
                return None
        if judgment is not None and judge_rubric_version in (
            RUBRIC_VERSION_V0_9,
            RUBRIC_VERSION_V0_10,
            RUBRIC_VERSION_V0_11,
            RUBRIC_VERSION_V0_12,
            RUBRIC_VERSION_V0_13,
            RUBRIC_VERSION_V0_14,
            RUBRIC_VERSION_V0_15,
            RUBRIC_VERSION_V0_16,
            RUBRIC_VERSION_V0_17,
            RUBRIC_VERSION_V0_18,
            RUBRIC_VERSION_V0_19,
            RUBRIC_VERSION_V0_20,
            RUBRIC_VERSION_V0_21,
            RUBRIC_VERSION_V0_22,
            RUBRIC_VERSION_V0_23,
            RUBRIC_VERSION_V0_24,
            RUBRIC_VERSION_V0_25,
            RUBRIC_VERSION_V0_26,
            RUBRIC_VERSION_V0_27,
        ):
            if not v0_8_evidence_contract_valid(judgment):
                return None
            source_case = selected_by_case_id.get(str(info.get("case_id")))
            if source_case is None or not evidence_quotes_match_trace(
                judgment,
                source_case.get("candidate_trace", []),
                allow_normalized_tool_calls=judge_rubric_version in (
                    RUBRIC_VERSION_V0_15,
                    RUBRIC_VERSION_V0_16,
                    RUBRIC_VERSION_V0_17,
                    RUBRIC_VERSION_V0_18,
                    RUBRIC_VERSION_V0_19,
                    RUBRIC_VERSION_V0_20,
                    RUBRIC_VERSION_V0_21,
                    RUBRIC_VERSION_V0_22,
                    RUBRIC_VERSION_V0_23,
                    RUBRIC_VERSION_V0_24,
                    RUBRIC_VERSION_V0_25,
                    RUBRIC_VERSION_V0_26,
                    RUBRIC_VERSION_V0_27,
                ),
                allow_smiles_tool_call_fragment=judge_rubric_version in (
                    RUBRIC_VERSION_V0_16,
                    RUBRIC_VERSION_V0_17,
                    RUBRIC_VERSION_V0_18,
                    RUBRIC_VERSION_V0_19,
                    RUBRIC_VERSION_V0_20,
                    RUBRIC_VERSION_V0_21,
                    RUBRIC_VERSION_V0_22,
                    RUBRIC_VERSION_V0_23,
                    RUBRIC_VERSION_V0_24,
                    RUBRIC_VERSION_V0_25,
                    RUBRIC_VERSION_V0_26,
                    RUBRIC_VERSION_V0_27,
                ),
                allow_ordered_ellipsis_fragments=judge_rubric_version in (
                    RUBRIC_VERSION_V0_22,
                    RUBRIC_VERSION_V0_23,
                    RUBRIC_VERSION_V0_24,
                    RUBRIC_VERSION_V0_25,
                    RUBRIC_VERSION_V0_26,
                    RUBRIC_VERSION_V0_27,
                ),
            ):
                return None
            if judge_rubric_version == RUBRIC_VERSION_V0_27:
                proof_validator = v0_27_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_26:
                proof_validator = v0_22_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_25:
                proof_validator = v0_22_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_24:
                proof_validator = v0_22_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_23:
                proof_validator = v0_22_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_22:
                proof_validator = v0_22_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_21:
                proof_validator = v0_21_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_20:
                proof_validator = v0_19_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_19:
                proof_validator = v0_19_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_18:
                proof_validator = v0_18_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_17:
                # The prompt makes the semantic analogy/substitution decision;
                # this guard verifies that the cited neighbor is actually in
                # the tool result without requiring one specific alias phrase.
                proof_validator = v0_17_proof_semantics_valid
            elif judge_rubric_version in (
                RUBRIC_VERSION_V0_14,
                RUBRIC_VERSION_V0_15,
                RUBRIC_VERSION_V0_16,
            ):
                proof_validator = v0_14_proof_semantics_valid
            elif judge_rubric_version == RUBRIC_VERSION_V0_13:
                proof_validator = v0_13_proof_semantics_valid
            else:
                proof_validator = v0_9_proof_semantics_valid
            if not proof_validator(judgment, source_case.get("candidate_trace", [])):
                return None
        return judgment

    def judge_label_correct(completion: vf.Messages, answer: str, info: dict[str, Any], **_: Any) -> float:
        judgment = parsed_judgment(completion, info)
        if judgment is None:
            return 0.0
        gold = parse_gold(answer)
        return float(judgment.valid_strategy == gold["valid_strategy"])

    def json_parse_success(completion: vf.Messages, info: dict[str, Any], **_: Any) -> float:
        return float(parsed_judgment(completion, info) is not None)

    def violation_set_f1(completion: vf.Messages, answer: str, info: dict[str, Any], **_: Any) -> float:
        judgment = parsed_judgment(completion, info)
        if judgment is None:
            return 0.0
        gold = parse_gold(answer)
        return violation_f1(judgment.violations, gold["violations"])

    def predicted_invalid(completion: vf.Messages, info: dict[str, Any], **_: Any) -> float:
        judgment = parsed_judgment(completion, info)
        return float(judgment is not None and not judgment.valid_strategy)

    def raw_predicted_invalid(completion: vf.Messages, **_: Any) -> float:
        judgment = parse_judgment(completion)
        return float(judgment is not None and not judgment.valid_strategy)

    def invalid_proof_rejected(completion: vf.Messages, info: dict[str, Any], **_: Any) -> float:
        judgment = parse_judgment(completion)
        return float(
            judgment is not None
            and not judgment.valid_strategy
            and parsed_judgment(completion, info) is None
        )

    rubric = vf.Rubric(
        funcs=[
            judge_label_correct,
            json_parse_success,
            violation_set_f1,
            predicted_invalid,
            raw_predicted_invalid,
            invalid_proof_rejected,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
