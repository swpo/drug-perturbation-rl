"""On-policy runtime for the frozen continuous process-coherence judge.

The deterministic phenotype rubric remains the sole biological-correctness
scorer.  This module audits whether each task-aware reasoning node is supported
and whether the directed transition is coherent.  The language model emits
only discrete labels; frozen local code validates cited message locations and
maps those labels to a score in ``[0, 1]``.  Every technical or evidence-proof
failure fails open to ``1.0`` and is exposed separately.
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

from phenotype_process_judge.contract import (
    NODE_STATUS_SCORES,
    PROCESS_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    TRANSITION_STATUS_SCORES,
    ProcessJudgment,
    evaluate_completion,
    task_graph,
)


PROCESS_JUDGE_MODES = {"off", "multiply"}


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


def _evidence_locations(
    original_prompt: list[dict[str, Any]],
    candidate_trace: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
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
            ("prompt", original_prompt),
            ("trace", candidate_trace),
        )
    }


def _judgment_event_payload(judgment: ProcessJudgment | None) -> dict[str, Any] | None:
    if judgment is None:
        return None
    return {
        "nodes": [
            {
                "name": node.name,
                "status": node.status,
                "evidence": [
                    {
                        "source": ref.source,
                        "message_index": ref.message_index,
                        "field": ref.field,
                    }
                    for ref in node.evidence
                ],
            }
            for node in judgment.nodes
        ],
        "transitions": [
            {
                "from": edge.source_node,
                "to": edge.target_node,
                "status": edge.status,
                "evidence": [
                    {
                        "source": ref.source,
                        "message_index": ref.message_index,
                        "field": ref.field,
                    }
                    for ref in edge.evidence
                ],
            }
            for edge in judgment.transitions
        ],
        "summary": judgment.summary,
    }


@dataclass
class ProcessJudgeRuntime:
    """One frozen Gemini process judge shared by all reward calls in an env."""

    model: str
    base_url: str
    api_key: str
    team_id: str | None = None
    concurrency: int = 32
    max_tokens: int = 900
    max_retries: int = 2
    timeout_seconds: float = 180.0
    rubric_version: str = RUBRIC_VERSION
    client: AsyncOpenAI = field(init=False, repr=False)
    semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rubric_version != RUBRIC_VERSION:
            raise ValueError(
                f"The process judge is locked to {RUBRIC_VERSION}; "
                f"got {self.rubric_version!r}"
            )
        if self.concurrency < 1:
            raise ValueError("process_judge_concurrency must be positive")
        headers = {"X-Prime-Team-ID": self.team_id} if self.team_id else None
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=headers,
            max_retries=self.max_retries,
            timeout=self.timeout_seconds,
        )
        self.semaphore = asyncio.Semaphore(self.concurrency)

    @staticmethod
    def _initialize_state(
        state: dict[str, Any],
        *,
        trace_hash: str,
        entry_point: str,
    ) -> None:
        state.update(
            {
                "process_judge_trace_sha256": trace_hash,
                "process_judge_entry_point": entry_point,
                "process_judge_score": 1.0,
                "process_judge_node_score": 1.0,
                "process_judge_transition_score": 1.0,
                "process_judge_parse_success": 0.0,
                "process_judge_reference_valid": 0.0,
                "process_judge_fail_open": 0.0,
                "process_judge_failure": 0.0,
                "process_judge_support_adjustments": 0.0,
                "process_judge_latency_seconds": 0.0,
                "process_judge_input_tokens": 0.0,
                "process_judge_output_tokens": 0.0,
                **{
                    f"process_judge_node_{status}_fraction": 0.0
                    for status in NODE_STATUS_SCORES
                },
                **{
                    f"process_judge_transition_{status}_fraction": 0.0
                    for status in TRANSITION_STATUS_SCORES
                },
            }
        )

    @staticmethod
    def _record_label_fractions(
        state: dict[str, Any],
        judgment: ProcessJudgment,
    ) -> None:
        node_count = max(len(judgment.nodes), 1)
        transition_count = max(len(judgment.transitions), 1)
        for status in NODE_STATUS_SCORES:
            state[f"process_judge_node_{status}_fraction"] = (
                sum(node.status == status for node in judgment.nodes) / node_count
            )
        for status in TRANSITION_STATUS_SCORES:
            state[f"process_judge_transition_{status}_fraction"] = (
                sum(edge.status == status for edge in judgment.transitions)
                / transition_count
            )

    async def evaluate(
        self,
        *,
        prompt: Any,
        completion: Any,
        entry_point: str,
        state: dict[str, Any],
    ) -> float:
        """Audit and score one task-aware trace; return a fail-open multiplier."""
        original_prompt = _messages_to_dicts(prompt)
        candidate_trace = _messages_to_dicts(completion)
        trace_hash = _trace_sha256(original_prompt, candidate_trace)
        self._initialize_state(
            state,
            trace_hash=trace_hash,
            entry_point=entry_point,
        )

        try:
            graph = task_graph(entry_point)
        except Exception as exc:
            state["process_judge_failure"] = 1.0
            state["process_judge_fail_open"] = 1.0
            state["process_judge_error_type"] = type(exc).__name__
            state["process_judge_error"] = str(exc)
            self._emit_event(state, judgment=None, raw_response=None)
            return 1.0

        case_payload = {
            "case_id": trace_hash,
            "entry_point": entry_point,
            "task_graph": graph,
            "evidence_locations": _evidence_locations(
                original_prompt,
                candidate_trace,
            ),
            "original_prompt": original_prompt,
            "candidate_trace": candidate_trace,
        }
        user_prompt = (
            "Audit this candidate's reasoning process. "
            "Return only the required JSON.\n\n"
            "<candidate_case>\n"
            + json.dumps(case_payload, ensure_ascii=False, indent=2)
            + "\n</candidate_case>"
        )

        raw_response: str | None = None
        started = time.monotonic()
        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PROCESS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
            raw_response = _response_text(response)
            usage = getattr(response, "usage", None)
            state["process_judge_input_tokens"] = float(
                _usage_value(usage, "prompt_tokens", "input_tokens")
            )
            state["process_judge_output_tokens"] = float(
                _usage_value(usage, "completion_tokens", "output_tokens")
            )
        except Exception as exc:  # fail-open is intentional for judge infrastructure
            state["process_judge_failure"] = 1.0
            state["process_judge_fail_open"] = 1.0
            state["process_judge_error_type"] = type(exc).__name__
            state["process_judge_error"] = str(exc)
            self._emit_event(state, judgment=None, raw_response=None)
            return 1.0
        finally:
            state["process_judge_latency_seconds"] = time.monotonic() - started

        state["process_judge_raw_response"] = raw_response
        evaluation = evaluate_completion(
            [{"role": "assistant", "content": raw_response or ""}],
            original_prompt=original_prompt,
            candidate_trace=candidate_trace,
            expected_nodes=graph["audited_nodes"],
            generated_nodes=graph["generated_nodes"],
            expected_transitions=graph["transitions"],
        )
        state["process_judge_parse_success"] = float(evaluation.parse_success)
        state["process_judge_reference_valid"] = float(evaluation.proof_valid)
        state["process_judge_support_adjustments"] = float(
            evaluation.support_adjustments
        )
        if evaluation.judgment is not None:
            self._record_label_fractions(state, evaluation.judgment)

        if not evaluation.parse_success:
            state["process_judge_fail_open"] = 1.0
            state["process_judge_error_type"] = "JudgeParseError"
        elif not evaluation.proof_valid:
            state["process_judge_fail_open"] = 1.0
            state["process_judge_error_type"] = "JudgeReferenceError"

        state["process_judge_score"] = float(evaluation.process_score)
        state["process_judge_node_score"] = float(evaluation.node_score)
        state["process_judge_transition_score"] = float(
            evaluation.transition_score
        )
        self._emit_event(
            state,
            judgment=evaluation.judgment,
            raw_response=raw_response,
        )
        return float(state["process_judge_score"])

    @staticmethod
    def _emit_event(
        state: dict[str, Any],
        *,
        judgment: ProcessJudgment | None,
        raw_response: str | None,
    ) -> None:
        """Emit compact labels/references for every call and raw text on failure."""
        event = {
            "event": "phenotype_process_judge",
            "rubric_version": RUBRIC_VERSION,
            "trace_sha256": state.get("process_judge_trace_sha256"),
            "entry_point": state.get("process_judge_entry_point"),
            "process_score": state.get("process_judge_score", 1.0),
            "node_score": state.get("process_judge_node_score", 1.0),
            "transition_score": state.get(
                "process_judge_transition_score",
                1.0,
            ),
            "parse_success": state.get("process_judge_parse_success", 0.0),
            "reference_valid": state.get("process_judge_reference_valid", 0.0),
            "fail_open": state.get("process_judge_fail_open", 0.0),
            "failure": state.get("process_judge_failure", 0.0),
            "support_adjustments": state.get(
                "process_judge_support_adjustments",
                0.0,
            ),
            "judgment": _judgment_event_payload(judgment),
            "error_type": state.get("process_judge_error_type"),
            "error": state.get("process_judge_error"),
        }
        if judgment is None and raw_response is not None:
            event["raw_response"] = raw_response
        print("PROCESS_JUDGE_EVENT " + json.dumps(event, ensure_ascii=False), flush=True)


def build_process_judge_runtime(
    *,
    mode: str,
    model: str,
    base_url: str,
    api_key_var: str,
    concurrency: int,
    max_tokens: int,
    max_retries: int,
    timeout_seconds: float,
) -> ProcessJudgeRuntime | None:
    """Construct the judge only for continuous-multiplier mode."""
    if mode not in PROCESS_JUDGE_MODES:
        raise ValueError(
            f"process_judge_mode must be one of {sorted(PROCESS_JUDGE_MODES)}, "
            f"got {mode!r}"
        )
    if mode == "off":
        return None
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise ValueError(f"Missing required process judge key: {api_key_var}")
    return ProcessJudgeRuntime(
        model=model,
        base_url=base_url,
        api_key=api_key,
        team_id=os.environ.get("PRIME_TEAM_ID"),
        concurrency=concurrency,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
