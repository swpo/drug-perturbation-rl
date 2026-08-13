"""Offline integration checks for the v0.10.3 strategy-judge reward wiring."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from bioreasoning_phenotype.env import load_examples, make_rubric
from bioreasoning_phenotype.strategy_judge import StrategyJudgeRuntime


@dataclass
class _FixedGate:
    value: float

    async def evaluate(
        self,
        *,
        prompt: Any,
        completion: Any,
        state: dict[str, Any],
    ) -> float:
        del prompt, completion
        state.update(
            {
                "strategy_judge_parse_success": 1.0,
                "strategy_judge_raw_invalid": float(self.value == 0.0),
                "strategy_judge_operational_gate": self.value,
                "strategy_judge_proof_accepted_invalid": float(self.value == 0.0),
                "strategy_judge_proof_rejected_invalid": 0.0,
                "strategy_judge_failure": 0.0,
                "strategy_judge_latency_seconds": 0.01,
                "strategy_judge_input_tokens": 100.0,
                "strategy_judge_output_tokens": 20.0,
            }
        )
        return self.value


class _FailingCompletions:
    async def create(self, **_: Any) -> Any:
        raise RuntimeError("synthetic judge transport failure")


class _FailingChat:
    completions = _FailingCompletions()


class _FailingClient:
    chat = _FailingChat()


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
            "content": "<TARGET>" + " | ".join(answer["target"]) + "</TARGET>",
        }
    ]
    return {
        "prompt": row["prompt"],
        "completion": completion,
        "answer": row["answer"],
        "info": row["info"],
        "input": row,
        "trajectory": [],
    }


async def _check_reward_wiring() -> None:
    shadow_state = _perfect_target_state()
    shadow = make_rubric(
        strategy_judge=_FixedGate(0.0),
        strategy_judge_mode="shadow",
    )
    await shadow.score_group([shadow_state])
    assert shadow_state["metrics"]["aggregate_reward"] == 1.0
    assert shadow_state["metrics"]["strategy_judge_operational_gate"] == 0.0
    assert shadow_state["metrics"]["training_reward"] == 1.0
    assert shadow_state["reward"] == 1.0

    gate_state = _perfect_target_state()
    gate = make_rubric(
        strategy_judge=_FixedGate(0.0),
        strategy_judge_mode="gate",
    )
    await gate.score_group([gate_state])
    assert gate_state["metrics"]["aggregate_reward"] == 1.0
    assert gate_state["metrics"]["strategy_judge_operational_gate"] == 0.0
    assert gate_state["metrics"]["training_reward"] == 0.0
    assert gate_state["reward"] == 0.0


async def _check_transport_failure_fails_open() -> None:
    runtime = StrategyJudgeRuntime(
        model="synthetic",
        base_url="https://example.invalid/v1",
        api_key="synthetic",
        concurrency=1,
        max_tokens=10,
        max_retries=0,
        timeout_seconds=1.0,
    )
    runtime.client = _FailingClient()  # type: ignore[assignment]
    state: dict[str, Any] = {}
    gate = await runtime.evaluate(
        prompt=[{"role": "user", "content": "test"}],
        completion=[{"role": "assistant", "content": "answer"}],
        state=state,
    )
    assert gate == 1.0
    assert state["strategy_judge_operational_gate"] == 1.0
    assert state["strategy_judge_failure"] == 1.0
    assert state["strategy_judge_error_type"] == "RuntimeError"


async def main() -> int:
    await _check_reward_wiring()
    await _check_transport_failure_fails_open()
    print("PASS: shadow/gate wiring and judge fail-open behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
