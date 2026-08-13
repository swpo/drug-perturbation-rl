"""Contract and metamorphic-construction tests for reasoning-process-v0.3."""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "judges" / "phenotype_process_judge"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_process_judge_corpus import build_rows, read_jsonl  # noqa: E402
from phenotype_process_judge.contract import (  # noqa: E402
    PROCESS_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    evaluate_completion,
    infer_entry_point,
    parse_process_judgment,
    task_graph,
)


SOURCE = Path(os.environ.get("PROCESS_JUDGE_SOURCE_CORPUS", "__missing__"))


def completion(payload: dict) -> list[dict]:
    return [{"role": "assistant", "content": json.dumps(payload)}]


class ProcessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = [
            {"role": "system", "content": "Reason carefully."},
            {
                "role": "user",
                "content": "Known protein target(s): BRAF\n<MOA>...</MOA>",
            },
        ]
        self.trace = [
            {
                "role": "assistant",
                "reasoning_content": "BRAF is a RAF kinase, so blocking it is RAF inhibition.",
                "content": "<MOA>RAF inhibitor</MOA>",
            }
        ]
        self.graph = task_graph("moa_from_target")
        self.payload = {
            "rubric_version": RUBRIC_VERSION,
            "nodes": [
                {
                    "name": "target",
                    "status": "supported",
                    "evidence": [
                        {
                            "source": "prompt",
                            "message_index": 1,
                            "field": "content",
                        }
                    ],
                },
                {
                    "name": "moa",
                    "status": "supported",
                    "evidence": [
                        {
                            "source": "trace",
                            "message_index": 0,
                            "field": "reasoning_content",
                        }
                    ],
                }
            ],
            "transitions": [
                {
                    "from": "target",
                    "to": "moa",
                    "status": "weak",
                    "evidence": [
                        {
                            "source": "trace",
                            "message_index": 0,
                            "field": "reasoning_content",
                        }
                    ],
                }
            ],
            "summary": "The answer has a relevant but compact bridge.",
        }

    def test_prompt_defines_process_not_hidden_correctness(self) -> None:
        self.assertIn("NOT grading the answer against hidden", PROCESS_SYSTEM_PROMPT)
        self.assertIn("Do not penalize brevity", PROCESS_SYSTEM_PROMPT)
        self.assertIn("emit a numeric score", PROCESS_SYSTEM_PROMPT)

    def test_valid_complete_schema_parses_and_scores(self) -> None:
        result = evaluate_completion(
            completion(self.payload),
            original_prompt=self.prompt,
            candidate_trace=self.trace,
            expected_nodes=self.graph["audited_nodes"],
            generated_nodes=self.graph["generated_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertTrue(result.parse_success)
        self.assertTrue(result.proof_valid)
        self.assertAlmostEqual(result.node_score, 1.0)
        self.assertAlmostEqual(result.transition_score, 0.65)
        self.assertAlmostEqual(result.process_score, 0.65 ** (1.0 / 3.0))

    def test_malformed_output_fails_open(self) -> None:
        payload = dict(self.payload)
        payload["nodes"] = []
        result = evaluate_completion(
            completion(payload),
            original_prompt=self.prompt,
            candidate_trace=self.trace,
            expected_nodes=self.graph["audited_nodes"],
            generated_nodes=self.graph["generated_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertFalse(result.parse_success)
        self.assertFalse(result.proof_valid)
        self.assertEqual(result.process_score, 1.0)

    def test_unproved_quote_fails_open(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["nodes"][0]["evidence"][0]["message_index"] = 99
        result = evaluate_completion(
            completion(payload),
            original_prompt=self.prompt,
            candidate_trace=self.trace,
            expected_nodes=self.graph["audited_nodes"],
            generated_nodes=self.graph["generated_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertTrue(result.parse_success)
        self.assertFalse(result.proof_valid)
        self.assertEqual(result.process_score, 1.0)

    def test_contradiction_maps_to_zero(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["transitions"][0]["status"] = "contradicted"
        result = evaluate_completion(
            completion(payload),
            original_prompt=self.prompt,
            candidate_trace=self.trace,
            expected_nodes=self.graph["audited_nodes"],
            generated_nodes=self.graph["generated_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertTrue(result.proof_valid)
        self.assertEqual(result.process_score, 0.0)

    def test_bare_answer_cannot_prove_supported_process(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["nodes"][1]["evidence"][0]["field"] = "content"
        payload["transitions"][0]["evidence"][0]["field"] = "content"
        result = evaluate_completion(
            completion(payload),
            original_prompt=self.prompt,
            candidate_trace=self.trace,
            expected_nodes=self.graph["audited_nodes"],
            generated_nodes=self.graph["generated_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertTrue(result.proof_valid)
        self.assertEqual(result.support_adjustments, 2)
        self.assertAlmostEqual(result.process_score, (0.25 * 0.25) ** (1.0 / 3.0))

    def test_extra_or_reordered_components_do_not_parse(self) -> None:
        payload = json.loads(json.dumps(self.payload))
        payload["nodes"].append(payload["nodes"][0])
        parsed = parse_process_judgment(
            completion(payload),
            expected_nodes=self.graph["audited_nodes"],
            expected_transitions=self.graph["transitions"],
        )
        self.assertIsNone(parsed)

    def test_direct_task_inference(self) -> None:
        for tag, expected in (
            ("<TARGET>x</TARGET>", "target_from_smiles"),
            ("<MOA>x</MOA>", "moa_from_target"),
            ("<PATHWAYS>x</PATHWAYS>", "pathways_from_moa"),
            ("<VIABILITY>x</VIABILITY>", "phenotype_from_pathways"),
        ):
            prompt = [{"role": "user", "content": f"Answer the requested task. {tag}"}]
            self.assertEqual(infer_entry_point(prompt), expected)


@unittest.skipUnless(SOURCE.exists(), "opened on-policy source corpus unavailable")
class MetamorphicCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_rows(read_jsonl(SOURCE))

    def test_balanced_variant_families(self) -> None:
        self.assertEqual(len(self.rows), 45)
        variants = {row["variant"] for row in self.rows}
        self.assertEqual(
            variants,
            {
                "original",
                "reasoning_deleted",
                "cross_prompt_reasoning",
                "explicit_disconnect",
                "irrelevant_annotation",
            },
        )
        for anchor in {row["anchor_id"] for row in self.rows}:
            self.assertEqual(sum(row["anchor_id"] == anchor for row in self.rows), 5)

    def test_answer_is_held_fixed_within_each_anchor(self) -> None:
        by_anchor: dict[str, list[dict]] = {}
        for row in self.rows:
            by_anchor.setdefault(row["anchor_id"], []).append(row)
        for rows in by_anchor.values():
            tags = []
            for row in rows:
                text = "\n".join(
                    str(message.get("content") or "")
                    for message in row["candidate_trace"]
                    if isinstance(message, dict) and message.get("role") == "assistant"
                )
                tags.append(
                    tuple(
                        line
                        for line in text.splitlines()
                        if line.startswith(("<TARGET>", "<MOA>", "<PATHWAYS>", "<VIABILITY>", "<CELL_CYCLE>", "<STRESS>"))
                    )
                )
            self.assertTrue(all(tag == tags[0] for tag in tags))

    def test_corruptions_are_relational_not_scalar_labels(self) -> None:
        for row in self.rows:
            self.assertNotIn("gold", row)
            if row["variant"] == "original":
                self.assertEqual(row["expectation"]["relation"], "baseline")
            elif row["variant"] == "irrelevant_annotation":
                self.assertEqual(row["expectation"]["relation"], "approximately_equal")
            else:
                self.assertEqual(row["expectation"]["relation"], "lower")


if __name__ == "__main__":
    unittest.main()
