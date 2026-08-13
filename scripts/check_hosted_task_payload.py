"""Check dataset rows against the current hosted task transport contract.

This canary guards the v0.10 baseline against reintroducing the legacy direct
string-valued ``task`` route. Newer Verifiers interprets ``task`` as a
serialized task *object*; environment routing belongs in ``info['env_id']``.
The single-environment phenotype dataset needs neither field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import verifiers as vf

from bioreasoning_phenotype import load_environment


def _check_row(row: Mapping[str, Any], *, split: str, index: int) -> None:
    where = f"{split}[{index}]"
    assert "task" not in row, f"{where}: direct task payload must be absent"

    info = row.get("info")
    assert isinstance(info, Mapping), f"{where}: info must be a JSON object"
    assert "task" not in info, f"{where}: info.task payload must be absent"

    prompt = row.get("prompt")
    assert isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)), (
        f"{where}: prompt must be a message sequence"
    )
    assert prompt, f"{where}: prompt must not be empty"
    assert all(isinstance(message, Mapping) for message in prompt), (
        f"{where}: every prompt message must be a JSON object"
    )

    # Match the transport's basic JSON gate before invoking newer helpers.
    json.dumps(dict(row))

    try:
        from verifiers.types import assert_json_serializable, flatten_task_input
    except ImportError:
        # These helpers were added after the original v0.10.1 dependency floor.
        return

    assert_json_serializable(dict(row))
    flattened = flatten_task_input(dict(row))
    assert flattened == dict(row), f"{where}: transport changed the task row"


def main() -> int:
    env = load_environment(
        entry_points=["smiles_only", "target_from_smiles"],
        phenotypes=["viability"],
        num_train_examples=16,
        num_eval_examples=16,
    )

    for split, dataset in (("train", env.dataset), ("eval", env.eval_dataset)):
        assert "task" not in dataset.column_names, (
            f"{split}: direct task column must be absent"
        )
        for index in range(len(dataset)):
            _check_row(dataset[index], split=split, index=index)

    print(
        "PASS: hosted task payload shape is valid "
        f"(verifiers {vf.__version__}; 32 rows checked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
