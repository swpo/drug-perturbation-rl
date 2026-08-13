# Research lineage

This repository presents a curated scientific line rather than the complete
chronology of every experiment in the original private workspace.

## Included

### Curriculum baseline (`0.10.1`)

The baseline is the four-link environment used for the legacy-fallback
curriculum and model-scaling experiments:

```text
SMILES → target → mechanism → signed pathways → phenotype
```

The curriculum combines full-chain examples with direct identity, mechanism,
and pathway refresh tasks. The historical reproduction configurations for the
2B, 4B, 9B, and 35B policies are in `configs/curriculum/`.

### Task mixture and filtering (`0.10.3`)

The next experiment replaces stage scheduling with a flat mixture of four
direct-link tasks. Groups with zero advantage are removed after batch assembly;
all other heuristic filters are diagnostic. A strategy judge can be measured in
shadow mode or applied as a proof-validated gate.

### Continuous process judge (`0.10.4`)

The process judge scores the validity and coherence of the candidate's reasoning
trace. Its score is constrained to `[0, 1]` and multiplies, rather than replaces
or increases, deterministic biological reward.

`configs/filtering/` and `configs/process/` include the matched 50-step pilot
recipes and independent scratch 300-step recipes. Operational continuation and
recovery launchers are omitted: they split one scientific trajectory around
hosted-platform checkpoint boundaries but do not define new experimental
conditions.

## Deliberately excluded

The original workspace also explored an explicit intermediate evidence bridge
and later pathway/response-model variants. Those experiments formed a side
branch between the curriculum and filtering work, were not dependencies of the
filtering experiments, and are omitted here.

Private operational memory, exhaustive experiment logs, raw service downloads,
and long internal reports are also omitted. Public-facing results will be added
as compact figures and methods summaries after review.
