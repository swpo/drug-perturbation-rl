"""Offline integration checks for the v0.10.4 process-judge reward wiring."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from bioreasoning_phenotype.env import load_examples, make_rubric
from bioreasoning_phenotype.process_judge import ProcessJudgeRuntime
from phenotype_process_judge.contract import RUBRIC_VERSION


@dataclass
class _FixedProcessScore:
    value: float

    async def evaluate(
        self,
        *,
        prompt: Any,
        completion: Any,
        entry_point: str,
        state: dict[str, Any],
    ) -> float:
        del prompt, completion
        assert entry_point == "target_from_smiles"
        state.update(
            {
                "process_judge_score": self.value,
                "process_judge_node_score": self.value,
                "process_judge_transition_score": self.value,
                "process_judge_parse_success": 1.0,
                "process_judge_reference_valid": 1.0,
                "process_judge_fail_open": 0.0,
                "process_judge_failure": 0.0,
                "process_judge_support_adjustments": 0.0,
                "process_judge_latency_seconds": 0.01,
                "process_judge_input_tokens": 100.0,
                "process_judge_output_tokens": 20.0,
            }
        )
        return self.value


class _FixedCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error

    async def create(self, **_: Any) -> Any:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content or ""),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


class _FixedClient:
    def __init__(self, completions: _FixedCompletions):
        self.chat = SimpleNamespace(completions=completions)


def _perfect_target_state() -> dict[str, Any]:
    train, _ = load_examples(
        entry_points=["target_from_smiles"],
        num_train_examples=1,
        num_eval_examples=1,
    )
    row = train[0]
    answer = json.loads(row["answer"])
    completion = [
        {
            "role": "assistant",
            "reasoning_content": (
                "The compound evidence supports the listed binding target."
            ),
            "content": "<TARGET>" + " | ".join(answer["target"]) + "</TARGET>",
        }
    ]
    return {
        "prompt": row["prompt"],
        "completion": completion,
        "answer": row["answer"],
        "info": row["info"],
        # Current Verifiers requires every scored state to carry task identity,
        # even when the reward functions under test do not consume it.
        "task": row,
        "input": row,
        "trajectory": [],
        "timing": {"total_ms": 0.0},
    }


async def _check_reward_wiring() -> None:
    off_state = _perfect_target_state()
    off = make_rubric()
    await off.score_group([off_state])
    assert off_state["metrics"]["aggregate_reward"] == 1.0
    assert off_state["metrics"]["training_reward"] == 1.0
    assert off_state["reward"] == 1.0

    process_state = _perfect_target_state()
    process = make_rubric(
        process_judge=_FixedProcessScore(0.65),
        process_judge_mode="multiply",
    )
    await process.score_group([process_state])
    assert process_state["metrics"]["aggregate_reward"] == 1.0
    assert process_state["metrics"]["process_judge_score"] == 0.65
    assert process_state["metrics"]["training_reward"] == 0.65
    assert process_state["reward"] == 0.65


def _runtime() -> ProcessJudgeRuntime:
    return ProcessJudgeRuntime(
        model="synthetic",
        base_url="https://example.invalid/v1",
        api_key="synthetic",
        concurrency=1,
        max_tokens=900,
        max_retries=0,
        timeout_seconds=1.0,
    )


async def _check_runtime_success() -> None:
    payload = {
        "rubric_version": RUBRIC_VERSION,
        "nodes": [
            {
                "name": "compound_evidence",
                "status": "supported",
                "evidence": [
                    {
                        "source": "prompt",
                        "message_index": 0,
                        "field": "content",
                    }
                ],
            },
            {
                "name": "target",
                "status": "supported",
                "evidence": [
                    {
                        "source": "trace",
                        "message_index": 0,
                        "field": "reasoning_content",
                    }
                ],
            },
        ],
        "transitions": [
            {
                "from": "compound_evidence",
                "to": "target",
                "status": "coherent",
                "evidence": [
                    {
                        "source": "trace",
                        "message_index": 0,
                        "field": "reasoning_content",
                    }
                ],
            }
        ],
        "summary": "The compound evidence coherently supports the target.",
    }
    runtime = _runtime()
    runtime.client = _FixedClient(_FixedCompletions(json.dumps(payload)))  # type: ignore[assignment]
    state: dict[str, Any] = {}
    score = await runtime.evaluate(
        prompt=[{"role": "user", "content": "Compound evidence."}],
        completion=[
            {
                "role": "assistant",
                "reasoning_content": "The evidence supports this target assignment.",
                "content": "<TARGET>BRAF</TARGET>",
            }
        ],
        entry_point="target_from_smiles",
        state=state,
    )
    assert score == 1.0
    assert state["process_judge_parse_success"] == 1.0
    assert state["process_judge_reference_valid"] == 1.0
    assert state["process_judge_fail_open"] == 0.0
    assert state["process_judge_node_supported_fraction"] == 1.0
    assert state["process_judge_transition_coherent_fraction"] == 1.0


async def _check_transport_failure_fails_open() -> None:
    runtime = _runtime()
    runtime.client = _FixedClient(  # type: ignore[assignment]
        _FixedCompletions(error=RuntimeError("synthetic transport failure"))
    )
    state: dict[str, Any] = {}
    score = await runtime.evaluate(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "answer"}],
        entry_point="target_from_smiles",
        state=state,
    )
    assert score == 1.0
    assert state["process_judge_score"] == 1.0
    assert state["process_judge_fail_open"] == 1.0
    assert state["process_judge_failure"] == 1.0
    assert state["process_judge_error_type"] == "RuntimeError"


def _check_mutual_exclusion() -> None:
    try:
        make_rubric(
            strategy_judge=object(),  # type: ignore[arg-type]
            strategy_judge_mode="gate",
            process_judge=object(),  # type: ignore[arg-type]
            process_judge_mode="multiply",
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually exclusive judge modes to fail")


async def main() -> int:
    await _check_reward_wiring()
    await _check_runtime_success()
    await _check_transport_failure_fails_open()
    _check_mutual_exclusion()
    print("PASS: continuous reward wiring, structured scoring, and fail-open behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
