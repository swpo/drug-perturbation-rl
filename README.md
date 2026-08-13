# Drug Perturbation RL

Reinforcement-learning environments and analysis tools for multi-step reasoning
about small-molecule perturbations. Given a molecular structure, cell line, and
assay context, a policy predicts a causal chain from protein targets through
mechanism of action and signed pathways to cellular phenotype.

The repository follows one intentionally narrow research lineage:

1. the four-stage curriculum environment used for the strongest curriculum
   results;
2. a flat mixture of direct-link tasks with zero-advantage filtering;
3. shadow and gated strategy judges; and
4. a continuous process judge that can only dampen deterministic reward.

Experimental evidence-bridge and response-model branches are deliberately not
part of this repository. See [Research lineage](docs/RESEARCH_LINEAGE.md).

## Reasoning environment

The environment grades four links independently:

| Link | Output | Score |
| --- | --- | --- |
| SMILES → target | gene-symbol set | set F1 |
| target → mechanism | mechanism of action | normalized exact match |
| mechanism → pathways | signed Hallmark pathways | set F1 |
| pathways → phenotype | viability, cell cycle, stress, or magnitude | task-specific score |

Full-chain prompts request all upstream links and one phenotype. Direct-link
entry points expose individual links for curriculum and task-mixture training.
The optional compound tool returns identity and chemistry information without
revealing biological outcomes.

## Layout

```text
bioreasoning_phenotype/  Environment implementation and packaged data
configs/                 Frozen hosted-training configurations
judges/                  Frozen strategy and process judge contracts
scripts/                 Data construction, smoke tests, and eval utilities
docs/                    Research lineage, data provenance, and reproduction notes
```

The repository name is broader than the historical import package. The Python
module remains `bioreasoning_phenotype` so that published environment artifacts
and frozen configurations remain reproducible.

## Install and test

Python 3.11 or newer is required.

```bash
uv sync --all-extras
uv run python scripts/smoke.py
```

The smoke test is offline: it checks packaged data, prompt construction, tool
behavior, and deterministic scoring without calling a model API.

Judge integration checks are also offline:

```bash
uv run python scripts/check_strategy_judge_runtime.py
uv run python scripts/check_process_judge_runtime.py
uv run python -m unittest discover -s tests
```

## Evaluation

Install the environment locally or use the historical Hub artifact:

```bash
uv run prime eval run abugoot/bioreasoning_phenotype@0.10.1 \
  -m openai/gpt-4.1-mini \
  -n 100 -r 8 --max-tokens 16384 \
  -a '{"entry_points":["smiles_only"],"phenotypes":["viability","cell_cycle","stress"],"tools":true,"hallmark_tools":false,"max_tool_turns":5}'
```

Frozen historical training configurations are provenance artifacts. Resume
configurations contain immutable checkpoint identifiers and are not intended as
generic launch templates.

## Data and licensing

The packaged examples are derived from LINCS L1000, DepMap PRISM, the Drug
Repurposing Hub, and MSigDB Hallmark gene sets. Those data retain their source
licenses and attribution requirements; they are not covered by the eventual
repository code license. See [Data sources](docs/DATA_SOURCES.md).

The code license is intentionally pending local review before a public remote is
created.

## Reproducibility

The initial environment snapshot corresponds to
`abugoot/bioreasoning_phenotype@0.10.1`. File hashes and the relationship between
the historical environment versions are documented in
[Reproducibility](docs/REPRODUCIBILITY.md).
