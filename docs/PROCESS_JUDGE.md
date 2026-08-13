# Continuous process judge

Version `0.10.4` adds a continuous reasoning-process score while preserving the
`0.10.3` tasks, prompts, data, deterministic biological scorers, tools, and
strategy-judge implementation.

## Reward contract

```text
training_reward = deterministic_reward * process_judge_score
0 <= process_judge_score <= 1
```

The process judge can only dampen deterministic reward. It cannot award credit
for an incorrect biological answer or increase a score above the deterministic
grader's output. Strategy-gate and process-multiplier modes are mutually
exclusive.

The judge audits task-aware reasoning nodes and transitions. It checks that:

- each generated claim has support in the candidate trace or supplied prompt;
- later claims build coherently on earlier claims; and
- cited evidence references an actual prompt or trace field.

Supported, weak, unsupported, and contradicted nodes map to `1.0`, `0.65`,
`0.25`, and `0.0`. Coherent, weak, disconnected, and contradicted transitions
use the same numeric map. The aggregate is a geometric mean over every audited
node and transition.

Schema, parse, evidence-reference, and transport failures fail open to a score
of `1.0` and are exposed through separate telemetry.

## Frozen runtime

- Rubric: `reasoning-process-v0.3`
- Historical judge: `google/gemini-3-flash-preview`
- Temperature: `0`
- Maximum output: `900` tokens
- Retries: `2`
- Timeout: `180` seconds
- Concurrency: `32` per environment process

## Development method

The rubric was tested relationally rather than by enumerating forbidden answer
styles. For each opened on-policy anchor, the final tagged answer was held fixed
while the reasoning was retained, removed, swapped across prompts, explicitly
disconnected, or harmlessly annotated. This checks whether the judge responds to
reasoning support and coherence rather than verbosity or answer identity.

`scripts/build_process_judge_corpus.py` reconstructs that paired corpus from an
explicitly supplied JSONL source. The source rollouts are not bundled.

## Files

- `bioreasoning_phenotype/process_judge.py`: on-policy runtime and telemetry.
- `judges/phenotype_process_judge/`: frozen task graph, schema, evidence checks,
  and continuous scoring contract.
- `configs/process/`: historical matched process-treatment configuration.
- `scripts/check_process_judge_runtime.py`: offline reward-wiring, schema, and
  fail-open checks.
- `tests/test_process_judge.py`: deterministic contract tests.
