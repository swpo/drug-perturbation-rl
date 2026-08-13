# Reproducibility

## Baseline environment

The initial source snapshot is the runtime published as:

- Environment: `abugoot/bioreasoning_phenotype@0.10.1`
- Environment ID: `l8ev8sc9az8kszt3rasg5jdb`
- Created: 2026-07-06 14:49:09 UTC
- Original private-workspace recovery commit: `6d55c1d`

The source was recovered from the immutable Hub artifact before the filtering
experiments began. It is the four-link curriculum baseline and does not contain
the later evidence/pathway side branch.

## Version map

| Historical version | Role in this repository |
| --- | --- |
| `0.10.1` | exact curriculum and standard-evaluation baseline |
| `0.10.2` | compatibility/provenance-only reconciliation; no scientific change |
| `0.10.3` | flat direct-link mixture, zero-advantage filtering, strategy judge |
| `0.10.4` | continuous process-judge multiplier |

The Git history is intentionally curated around those scientific transitions.
Historical version numbers are retained to map code to published environments;
they are not a claim that every intervening private experiment is present.

## Baseline checksums

The following hashes pin the packaged runtime assets from `0.10.1`:

```text
f462b5d581caaac83812fee88c9033e238e01f40a881554df8ea89a35a29de12  bioreasoning_phenotype/data/compound_table.parquet
7ba63fbfcf8ac46fd75344809491fdf46a7f4b80b2b58788d2765a89e7f030b0  bioreasoning_phenotype/data/hallmarks.json
9f1cedc98d93f84069d4ed823325adb6e03fca26c611ca42e11628d057b9c048  bioreasoning_phenotype/data/smallmol_chain_examples.parquet
dca6e43f531af4fbb9ee405b71e5c623c0e6782b837292825ef4578ac6bddf7a  bioreasoning_phenotype/env.py
91ae45cdda8d714033a864ee24a702ecc61b059e09c0da89ce9577ae10123043  bioreasoning_phenotype/main.py
d37441f07b7405a061e41dbbd9c76da75e242f5cfb0a9f0d0ddc4254d57886e5  bioreasoning_phenotype/prompts.py
9321e7091c1ad182299d45d5c6674551a3b65989cad91b7b6c0987d51634f8c1  bioreasoning_phenotype/tools.py
```

Run `uv run python scripts/smoke.py` after installation to validate the local
snapshot without model or network calls.
