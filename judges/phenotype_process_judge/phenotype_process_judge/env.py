"""Verifiers environment for offline process-coherence judge development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from .contract import (
    PROCESS_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    evaluate_completion,
    infer_entry_point,
    task_graph,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
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


def _build_record(row: dict[str, Any]) -> dict[str, Any]:
    entry_point = row.get("entry_point") or infer_entry_point(row["original_prompt"])
    graph = task_graph(str(entry_point))
    case_payload = {
        "case_id": row["case_id"],
        "entry_point": entry_point,
        "task_graph": graph,
        "evidence_locations": {
            source: [
                {
                    "message_index": index,
                    "role": message.get("role"),
                    "fields": [
                        field
                        for field in ("content", "reasoning_content")
                        if isinstance(message.get(field), str) and message[field].strip()
                    ],
                }
                for index, message in enumerate(messages)
                if isinstance(message, dict)
            ]
            for source, messages in (
                ("prompt", row["original_prompt"]),
                ("trace", row["candidate_trace"]),
            )
        },
        "original_prompt": row["original_prompt"],
        "candidate_trace": row["candidate_trace"],
    }
    prompt = (
        "Audit this candidate's reasoning process. Return only the required JSON.\n\n"
        "<candidate_case>\n"
        + json.dumps(case_payload, ensure_ascii=False, indent=2)
        + "\n</candidate_case>"
    )
    return {
        "prompt": [
            {"role": "system", "content": PROCESS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "answer": json.dumps(row.get("expectation") or {}, sort_keys=True),
        "info": {
            "case_id": row["case_id"],
            "anchor_id": row.get("anchor_id"),
            "variant": row.get("variant"),
            "entry_point": entry_point,
            "rubric_version": RUBRIC_VERSION,
            "audited_nodes": graph["audited_nodes"],
            "transitions": graph["transitions"],
        },
    }


def load_environment(
    corpus_path: str,
    case_ids: list[str] | None = None,
    max_examples: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    path = Path(corpus_path).expanduser().resolve()
    selected = _read_jsonl(path)
    if case_ids:
        allowed = set(case_ids)
        selected = [row for row in selected if row.get("case_id") in allowed]
    if max_examples is not None:
        selected = selected[:max_examples]
    if not selected:
        raise ValueError(f"No process-audit cases found in {path}")

    selected_by_id = {str(row["case_id"]): row for row in selected}
    dataset = Dataset.from_list([_build_record(row) for row in selected])

    def evaluation(completion: vf.Messages, info: dict[str, Any]):
        row = selected_by_id[str(info["case_id"])]
        graph = task_graph(str(info["entry_point"]))
        return evaluate_completion(
            completion,
            original_prompt=row["original_prompt"],
            candidate_trace=row["candidate_trace"],
            expected_nodes=graph["audited_nodes"],
            generated_nodes=graph["generated_nodes"],
            expected_transitions=graph["transitions"],
        )

    def process_score(completion: vf.Messages, info: dict[str, Any], **_: Any) -> float:
        return evaluation(completion, info).process_score

    def process_node_score(completion: vf.Messages, info: dict[str, Any], **_: Any) -> float:
        return evaluation(completion, info).node_score

    def process_transition_score(
        completion: vf.Messages,
        info: dict[str, Any],
        **_: Any,
    ) -> float:
        return evaluation(completion, info).transition_score

    def process_parse_success(
        completion: vf.Messages,
        info: dict[str, Any],
        **_: Any,
    ) -> float:
        return float(evaluation(completion, info).parse_success)

    def process_proof_valid(
        completion: vf.Messages,
        info: dict[str, Any],
        **_: Any,
    ) -> float:
        return float(evaluation(completion, info).proof_valid)

    def process_support_adjustments(
        completion: vf.Messages,
        info: dict[str, Any],
        **_: Any,
    ) -> float:
        return float(evaluation(completion, info).support_adjustments)

    rubric = vf.Rubric(
        funcs=[
            process_score,
            process_node_score,
            process_transition_score,
            process_parse_success,
            process_proof_valid,
            process_support_adjustments,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
