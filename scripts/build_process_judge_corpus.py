#!/usr/bin/env python3
"""Build paired metamorphic cases from concise real on-policy traces.

The development target is relational rather than a catalog of invalid cases:
the same final answer is held fixed while its reasoning is removed, replaced
with another prompt's reasoning, contradicted, or harmlessly annotated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ANCHORS = (
    "gate-s19-p664-r54",
    "gate-s21-p8211-r33",
    "gate-s41-p6283-r5",
    "gate-s48-p5653-r41",
    "gate-s15-p1576-r3",
    "gate-s30-p4360-r47",
    "gate-s15-p14472-r33",
    "gate-s21-p14778-r29",
    "gate-s38-p561-r7",
)
TAG_PATTERN = re.compile(
    r"<(?:TARGET|MOA|PATHWAYS|VIABILITY|CELL_CYCLE|STRESS)>.*?"
    r"</(?:TARGET|MOA|PATHWAYS|VIABILITY|CELL_CYCLE|STRESS)>",
    re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def entry_point(row: dict[str, Any]) -> str:
    text = "\n".join(str(message.get("content") or "") for message in row["original_prompt"])
    requested = text.rsplit("Answer the requested task.", 1)[-1]
    if "<TARGET>" in requested:
        return "target_from_smiles"
    if "<MOA>" in requested:
        return "moa_from_target"
    if "<PATHWAYS>" in requested:
        return "pathways_from_moa"
    return "phenotype_from_pathways"


def answer_tags(trace: list[dict[str, Any]]) -> str:
    tags: list[str] = []
    for message in trace:
        if isinstance(message, dict) and message.get("role") == "assistant":
            tags.extend(TAG_PATTERN.findall(str(message.get("content") or "")))
    if not tags:
        raise ValueError("Candidate trace has no final answer tag")
    return "\n".join(tags)


def reasoning_text(trace: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in trace:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for field in ("reasoning_content", "content"):
            value = message.get(field)
            if isinstance(value, str) and value:
                if field == "content":
                    value = TAG_PATTERN.sub("", value)
                if value.strip():
                    parts.append(value.strip())
    return "\n\n".join(parts)


def stripped_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "assistant", "content": answer_tags(row["candidate_trace"])}]


def swapped_trace(row: dict[str, Any], donor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "reasoning_content": reasoning_text(donor["candidate_trace"]),
            "content": answer_tags(row["candidate_trace"]),
        }
    ]


def contradicted_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = copy.deepcopy(row["candidate_trace"])
    for message in reversed(trace):
        if isinstance(message, dict) and message.get("role") == "assistant":
            field = "reasoning_content" if isinstance(message.get("reasoning_content"), str) else "content"
            prior = str(message.get(field) or "")
            message[field] = (
                prior
                + "\n\nThe supplied upstream premise does not support this conclusion; "
                "I am retaining the tagged answer as an arbitrary guess."
            )
            return trace
    raise ValueError("Candidate trace has no assistant message")


def annotated_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = copy.deepcopy(row["candidate_trace"])
    for message in reversed(trace):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = (
                str(message.get("content") or "")
                + "\n\nAdministrative note: the requested answer tags were preserved."
            )
            return trace
    raise ValueError("Candidate trace has no assistant message")


def build_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["case_id"]: row for row in source_rows}
    anchors = [by_id[case_id] for case_id in DEFAULT_ANCHORS]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in anchors:
        by_task[entry_point(row)].append(row)

    output: list[dict[str, Any]] = []
    for anchor in anchors:
        task = entry_point(anchor)
        peers = by_task[task]
        donor = peers[(peers.index(anchor) + 1) % len(peers)]
        if donor is anchor:
            raise ValueError(f"Need at least two anchors for {task}")
        variants = {
            "original": copy.deepcopy(anchor["candidate_trace"]),
            "reasoning_deleted": stripped_trace(anchor),
            "cross_prompt_reasoning": swapped_trace(anchor, donor),
            "explicit_disconnect": contradicted_trace(anchor),
            "irrelevant_annotation": annotated_trace(anchor),
        }
        for variant, trace in variants.items():
            expectation: dict[str, Any]
            if variant == "original":
                expectation = {"relation": "baseline"}
            elif variant == "irrelevant_annotation":
                expectation = {
                    "relation": "approximately_equal",
                    "reference_case_id": f"{anchor['case_id']}--original",
                    "tolerance": 0.15,
                }
            else:
                expectation = {
                    "relation": "lower",
                    "reference_case_id": f"{anchor['case_id']}--original",
                }
            output.append(
                {
                    "case_id": f"{anchor['case_id']}--{variant}",
                    "anchor_id": anchor["case_id"],
                    "variant": variant,
                    "entry_point": task,
                    "original_prompt": copy.deepcopy(anchor["original_prompt"]),
                    "candidate_trace": trace,
                    "expectation": expectation,
                    "source_model": anchor.get("source_model"),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="JSONL source containing opened on-policy candidate traces",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL for the paired metamorphic corpus",
    )
    args = parser.parse_args()
    rows = build_rows(read_jsonl(args.source))
    write_jsonl(args.output, rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "sha256": digest,
        "rows": len(rows),
        "anchors": len({row["anchor_id"] for row in rows}),
        "variants": dict(Counter(row["variant"] for row in rows)),
        "entry_points": dict(Counter(row["entry_point"] for row in rows)),
    }
    manifest_path = args.output.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
