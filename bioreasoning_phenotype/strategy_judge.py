"""On-policy wrapper for the frozen phenotype strategy judge.

The deterministic phenotype rubric remains the sole biological-correctness
scorer.  This module only evaluates whether the response used a valid strategy.
Raw judge INVALID decisions are allowed to gate reward only when the frozen
deterministic proof checks accept the cited evidence.  Transport, parsing, and
proof failures fail open and are exposed as separate numeric metrics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from phenotype_strategy_judge.env import _literal_audit_v0_26
from phenotype_strategy_judge.rubric import (
    RUBRIC_VERSION_V0_27,
    VIOLATIONS,
    evidence_quotes_match_trace,
    judge_system_prompt,
    normalize_evidence_content_alias,
    parse_judgment,
    v0_8_evidence_contract_valid,
    v0_27_proof_semantics_valid,
)


STRATEGY_JUDGE_MODES = {"off", "shadow", "gate"}


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert Verifiers/Pydantic messages to the frozen audit representation."""
    if isinstance(message, dict):
        return {key: value for key, value in message.items() if value is not None}
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    record: dict[str, Any] = {}
    for key in (
        "role",
        "content",
        "reasoning_content",
        "thinking_blocks",
        "tool_calls",
        "tool_call_id",
        "name",
    ):
        value = getattr(message, key, None)
        if value is not None:
            record[key] = value
    return record


def _messages_to_dicts(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    return [_message_to_dict(message) for message in messages]


def _trace_sha256(prompt: list[dict[str, Any]], trace: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"original_prompt": prompt, "candidate_trace": trace},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return content if isinstance(content, str) else ""


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return 0


@dataclass
class StrategyJudgeRuntime:
    """One frozen Gemini judge shared by all reward calls in an env process."""

    model: str
    base_url: str
    api_key: str
    team_id: str | None = None
    concurrency: int = 32
    max_tokens: int = 600
    max_retries: int = 2
    timeout_seconds: float = 180.0
    rubric_version: str = RUBRIC_VERSION_V0_27
    client: AsyncOpenAI = field(init=False, repr=False)
    semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rubric_version != RUBRIC_VERSION_V0_27:
            raise ValueError(
                "The on-policy pilot is locked to strategy-validity-v0.27; "
                f"got {self.rubric_version!r}"
            )
        if self.concurrency < 1:
            raise ValueError("strategy_judge_concurrency must be positive")
        headers = {"X-Prime-Team-ID": self.team_id} if self.team_id else None
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=headers,
            max_retries=self.max_retries,
            timeout=self.timeout_seconds,
        )
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def evaluate(
        self,
        *,
        prompt: Any,
        completion: Any,
        state: dict[str, Any],
    ) -> float:
        """Run and proof-check one judgment; return the fail-open gate."""
        original_prompt = _messages_to_dicts(prompt)
        candidate_trace = _messages_to_dicts(completion)
        trace_hash = _trace_sha256(original_prompt, candidate_trace)
        case_row = {
            "case_id": trace_hash,
            "original_prompt": original_prompt,
            "candidate_trace": candidate_trace,
        }
        case_payload = {
            **case_row,
            "literal_audit": _literal_audit_v0_26(case_row),
        }
        user_prompt = (
            "Audit the candidate trace below under the frozen strategy-validity rubric. "
            "Return only the required JSON object.\n\n"
            "<candidate_case>\n"
            + json.dumps(case_payload, ensure_ascii=False, indent=2)
            + "\n</candidate_case>"
        )

        state.update(
            {
                "strategy_judge_trace_sha256": trace_hash,
                "strategy_judge_parse_success": 0.0,
                "strategy_judge_raw_invalid": 0.0,
                "strategy_judge_operational_gate": 1.0,
                "strategy_judge_proof_accepted_invalid": 0.0,
                "strategy_judge_proof_rejected_invalid": 0.0,
                "strategy_judge_failure": 0.0,
                "strategy_judge_latency_seconds": 0.0,
                "strategy_judge_input_tokens": 0.0,
                "strategy_judge_output_tokens": 0.0,
                **{f"strategy_judge_violation_{name}": 0.0 for name in VIOLATIONS},
            }
        )

        started = time.monotonic()
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": judge_system_prompt(self.rubric_version)},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
            raw_response = _response_text(response)
            usage = getattr(response, "usage", None)
            state["strategy_judge_input_tokens"] = float(
                _usage_value(usage, "prompt_tokens", "input_tokens")
            )
            state["strategy_judge_output_tokens"] = float(
                _usage_value(usage, "completion_tokens", "output_tokens")
            )
        except Exception as exc:  # fail-open is intentional for judge infrastructure
            state["strategy_judge_failure"] = 1.0
            state["strategy_judge_error_type"] = type(exc).__name__
            state["strategy_judge_error"] = str(exc)
            self._emit_event(state, raw_response=None)
            return 1.0
        finally:
            state["strategy_judge_latency_seconds"] = time.monotonic() - started

        state["strategy_judge_raw_response"] = raw_response
        judgment = parse_judgment([{"role": "assistant", "content": raw_response}])
        if judgment is not None:
            judgment = normalize_evidence_content_alias(judgment)
        if judgment is None:
            state["strategy_judge_failure"] = 1.0
            state["strategy_judge_error_type"] = "JudgeParseError"
            self._emit_event(state, raw_response=raw_response)
            return 1.0

        state["strategy_judge_parse_success"] = 1.0
        for violation in judgment.violations:
            state[f"strategy_judge_violation_{violation}"] = 1.0
        if judgment.valid_strategy:
            return 1.0

        state["strategy_judge_raw_invalid"] = 1.0
        proof_accepted = (
            v0_8_evidence_contract_valid(judgment)
            and evidence_quotes_match_trace(
                judgment,
                candidate_trace,
                allow_normalized_tool_calls=True,
                allow_smiles_tool_call_fragment=True,
                allow_ordered_ellipsis_fragments=True,
            )
            and v0_27_proof_semantics_valid(judgment, candidate_trace)
        )
        if proof_accepted:
            state["strategy_judge_proof_accepted_invalid"] = 1.0
            state["strategy_judge_operational_gate"] = 0.0
        else:
            state["strategy_judge_proof_rejected_invalid"] = 1.0
        self._emit_event(state, raw_response=raw_response)
        return float(state["strategy_judge_operational_gate"])

    @staticmethod
    def _emit_event(state: dict[str, Any], raw_response: str | None) -> None:
        """Log only invalid/failure cases; hashes join them to retained rollouts."""
        event = {
            "event": "phenotype_strategy_judge",
            "rubric_version": RUBRIC_VERSION_V0_27,
            "trace_sha256": state.get("strategy_judge_trace_sha256"),
            "raw_invalid": state.get("strategy_judge_raw_invalid", 0.0),
            "operational_gate": state.get("strategy_judge_operational_gate", 1.0),
            "proof_accepted_invalid": state.get(
                "strategy_judge_proof_accepted_invalid", 0.0
            ),
            "proof_rejected_invalid": state.get(
                "strategy_judge_proof_rejected_invalid", 0.0
            ),
            "error_type": state.get("strategy_judge_error_type"),
            "error": state.get("strategy_judge_error"),
            "raw_response": raw_response,
        }
        print("STRATEGY_JUDGE_EVENT " + json.dumps(event, ensure_ascii=False), flush=True)


def build_strategy_judge_runtime(
    *,
    mode: str,
    model: str,
    base_url: str,
    api_key_var: str,
    concurrency: int,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: float,
) -> StrategyJudgeRuntime | None:
    """Construct the judge only for shadow/gated modes."""
    if mode not in STRATEGY_JUDGE_MODES:
        raise ValueError(
            f"strategy_judge_mode must be one of {sorted(STRATEGY_JUDGE_MODES)}, got {mode!r}"
        )
    if mode == "off":
        return None
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise ValueError(f"Missing required strategy judge key: {api_key_var}")
    return StrategyJudgeRuntime(
        model=model,
        base_url=base_url,
        api_key=api_key,
        team_id=os.environ.get("PRIME_TEAM_ID"),
        concurrency=concurrency,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
