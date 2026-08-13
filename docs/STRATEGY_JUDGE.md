# Strategy judge

Version `0.10.3` adds a frozen strategy-validity judge without changing the
underlying tasks, prompts, tools, examples, or deterministic biological scores.

## Task mixture

The task-mixture pilot samples four direct links at equal source rates:

1. `target_from_smiles`
2. `moa_from_target`
3. `pathways_from_moa`
4. `phenotype_from_pathways`

The phenotype task is restricted to viability, cell cycle, and stress to match
the standard end-to-end holdout. Because the hosted trainer supports at most
three environment blocks, the identity and mechanism datasets share one block
at ratio `0.50`; pathway and phenotype use separate `0.25` blocks.

Zero-advantage groups are removed after batch assembly. Gibberish, repetition,
and pre-assembly zero advantage are recorded as diagnostics rather than enforced
filters.

## Judge modes

Both judge modes compute the unchanged deterministic reward and evaluate the
complete candidate trace, including reasoning fields, tool calls, tool results,
and the final answer.

- `shadow`: record the judge result but train on deterministic reward.
- `gate`: train on `deterministic_reward * operational_gate`.

Only an `INVALID` judgment whose cited evidence passes deterministic proof
checks closes the gate. Provider, transport, parse, and proof-validation failures
fail open and are reported as separate metrics. The judge does not receive gold
answers, deterministic scores, advantages, model identity, or treatment label.

## Frozen runtime

- Rubric: `strategy-validity-v0.27`
- Historical judge: `google/gemini-3-flash-preview`
- Temperature: `0`
- Maximum output: `600` tokens
- Retries: `2`
- Timeout: `180` seconds
- Concurrency: `32` per environment process

The historical runtime uses an OpenAI-compatible endpoint configured through
`PRIME_API_KEY` and optionally `PRIME_TEAM_ID`. No credentials are stored in the
repository.

## Files

- `bioreasoning_phenotype/strategy_judge.py`: on-policy client, fail-open logic,
  telemetry, and gate integration.
- `judges/phenotype_strategy_judge/`: frozen rubric, trace audit, response schema,
  and proof validation.
- `configs/filtering/`: matched shadow/gate pilot configurations.
- `scripts/check_strategy_judge_runtime.py`: offline integration checks using a
  fake judge client.
