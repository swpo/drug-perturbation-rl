"""Expand chain GT into the env's examples parquet (v0.10, skinny + curriculum).

For each (compound × cell_line) with full upstream chain (target + MoA + pathways_signed)
AND at least one phenotype label, generate full-chain/phenotype examples.
Standalone upstream curriculum examples are generated once per compound-cell pair.

Each example is *skinny* — model predicts the upstream chain plus exactly ONE
downstream phenotype (viability, cell_cycle, stress, or magnitude), or predicts
one upstream curriculum target such as target, MoA, or pathways.

Output: bioreasoning_phenotype/data/smallmol_chain_examples.parquet
"""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from bioreasoning_phenotype.prompts import (
    ENTRY_POINTS, PHENOTYPES, build_chain_prompt,
)

CACHE = Path("/tmp/bioreasoning_smallmol_data")
TEST_FRACTION = 0.10
HASH_SEED = "smallmol_chain_v3"


def split_for_compound(compound: str) -> str:
    h = int(hashlib.md5(f"{HASH_SEED}:{compound}".encode()).hexdigest(), 16)
    return "test" if (h % 10000) / 10000.0 < TEST_FRACTION else "train"


PHENOTYPE_GT_KEY = {
    "viability":  "viability_lfc",
    "cell_cycle": "cell_cycle",
    "stress":     "stress",
    "magnitude":  "magnitude",
}


def _nonempty(x):
    try:
        return len(x) > 0
    except TypeError:
        return False


def _requested_steps(conf: dict) -> list[str]:
    if "asks" in conf:
        return list(conf["asks"])
    if conf.get("skip_chain", False):
        return ["phenotype"]
    return [step for step in ("target", "moa", "pathways") if step not in conf["given"]] + ["phenotype"]


def _given_values(conf: dict, r: pd.Series, pathways_str: str) -> dict[str, str]:
    given = {}
    if "target" in conf["given"]:
        given["target"] = " | ".join(r["target"])
    if "moa" in conf["given"]:
        given["moa"] = r["moa"]
    if "pathways" in conf["given"]:
        given["pathways"] = pathways_str
    return given


def _protocol_kwargs(r: pd.Series) -> dict:
    """Assay context fields passed into prompt generation."""
    kwargs = {}
    if "prism_dose_um" in r and pd.notna(r["prism_dose_um"]):
        kwargs["prism_dose_um"] = r["prism_dose_um"]
    if "prism_duration_h" in r and pd.notna(r["prism_duration_h"]):
        kwargs["prism_duration_h"] = r["prism_duration_h"]
    return kwargs


def _protocol_columns(r: pd.Series) -> dict:
    """Extra provenance columns to carry in packaged examples."""
    screens = r.get("prism_screens")
    if isinstance(screens, (list, tuple)):
        screens = "|".join(str(s) for s in screens)
    doses = r.get("prism_doses_um")
    if isinstance(doses, (list, tuple)):
        doses = "|".join(str(d) for d in doses)
    return {
        "lincs_dose_um": r.get("dose_um"),
        "lincs_time_h": r.get("time_h"),
        "prism_dose_um": r.get("prism_dose_um"),
        "prism_doses_um": doses,
        "prism_duration_h": r.get("prism_duration_h"),
        "prism_screens": screens,
        "prism_n_full_ids": r.get("prism_n_full_ids"),
        "prism_aggregation": r.get("prism_aggregation"),
    }


def _answer_for_steps(
    steps: list[str],
    upstream_gt: dict,
    phenotype: str | None = None,
    phen_val=None,
) -> dict:
    gt_for_eval = {}
    if "target" in steps:
        gt_for_eval["target"] = upstream_gt["target"]
    if "moa" in steps:
        gt_for_eval["moa"] = upstream_gt["moa"]
    if "pathways" in steps:
        gt_for_eval["pathways_signed"] = upstream_gt["pathways_signed"]
    if "phenotype" in steps:
        if phenotype is None:
            raise ValueError("phenotype step requested without a phenotype")
        gt_key = PHENOTYPE_GT_KEY[phenotype]
        gt_for_eval["phenotype"] = phenotype
        gt_for_eval[gt_key] = (
            float(phen_val) if phenotype == "viability" else str(phen_val)
        )
    return gt_for_eval


def main(cache_dir: Path = CACHE) -> tuple[Path, Path]:
    print("Building chain examples (v0.10 skinny + curriculum)...")
    gt = pd.read_parquet(cache_dir / "chain_gt.parquet")
    print(f"  Loaded chain_gt: {len(gt)} (compound × cell_line) pairs")

    # Require all UPSTREAM scales present for the example to be valid
    has_upstream = (
        gt["target"].apply(_nonempty)
        & gt["moa"].notna()
        & gt["pathways_signed"].apply(_nonempty)
    )
    upstream = gt[has_upstream].reset_index(drop=True)
    print(f"  Pairs with full upstream (target+moa+pathways_signed): {len(upstream)}")

    rows = []
    skipped_no_phenotype = 0
    for _, r in upstream.iterrows():
        compound = r["compound"]
        cell_line = r["cell_line"]
        split = split_for_compound(compound)

        # Format the pathway list once for prompting + grading
        pathways_pairs = [(name, direction) for name, direction in r["pathways_signed"]]
        pathways_str = ", ".join(f"{name}:{direction}" for name, direction in pathways_pairs)

        upstream_gt = {
            "target":          list(r["target"]),
            "moa":             r["moa"],
            "pathways_signed": [list(p) for p in pathways_pairs],
        }
        protocol = _protocol_columns(r)
        protocol_kwargs = _protocol_kwargs(r)

        # Standalone upstream curriculum tasks are generated once per
        # compound-cell pair, not once per phenotype label.
        for entry_point, conf in ENTRY_POINTS.items():
            steps = _requested_steps(conf)
            if "phenotype" in steps:
                continue
            given = _given_values(conf, r, pathways_str)
            user_prompt = build_chain_prompt(
                smiles=r["smiles"],
                cell_line=cell_line,
                dose_um=r["dose_um"],
                timepoint_h=r["time_h"],
                phenotype="viability",
                given_scales=given,
                requested_steps=steps,
                **protocol_kwargs,
            )
            rows.append({
                "compound": compound,
                "cell_line": cell_line,
                "entry_point": entry_point,
                "phenotype": "upstream",
                "user_prompt": user_prompt,
                "answer_json": json.dumps(_answer_for_steps(steps, upstream_gt)),
                "split": split,
                **protocol,
            })

        for phenotype in PHENOTYPES:
            gt_key = PHENOTYPE_GT_KEY[phenotype]
            phen_val = r[gt_key]
            if phen_val is None or (isinstance(phen_val, float) and pd.isna(phen_val)):
                skipped_no_phenotype += 1
                continue

            for entry_point, conf in ENTRY_POINTS.items():
                steps = _requested_steps(conf)
                if "phenotype" not in steps:
                    continue

                given = _given_values(conf, r, pathways_str)

                skip_chain = conf.get("skip_chain", False)
                requested_steps = steps if "asks" in conf else None
                user_prompt = build_chain_prompt(
                    smiles=r["smiles"],
                    cell_line=cell_line,
                    dose_um=r["dose_um"],
                    timepoint_h=r["time_h"],
                    phenotype=phenotype,
                    given_scales=given,
                    skip_chain=skip_chain,
                    requested_steps=requested_steps,
                    **protocol_kwargs,
                )

                rows.append({
                    "compound": compound,
                    "cell_line": cell_line,
                    "entry_point": entry_point,
                    "phenotype": phenotype,
                    "user_prompt": user_prompt,
                    "answer_json": json.dumps(_answer_for_steps(steps, upstream_gt, phenotype, phen_val)),
                    "split": split,
                    **protocol,
                })

    df = pd.DataFrame(rows)
    print(f"\nBuilt {len(df)} examples ({skipped_no_phenotype} skipped for missing phenotype label)")
    print(f"  per entry_point: {df.entry_point.value_counts().to_dict()}")
    print(f"  per phenotype:   {df.phenotype.value_counts().to_dict()}")
    print(f"  per split:       {df.split.value_counts().to_dict()}")

    pkg_data = PACKAGE_ROOT / "bioreasoning_phenotype" / "data"
    pkg_data.mkdir(parents=True, exist_ok=True)
    pkg_out = pkg_data / "smallmol_chain_examples.parquet"
    cache_out = cache_dir / "smallmol_chain_examples.parquet"
    df.to_parquet(pkg_out, index=False)
    df.to_parquet(cache_out, index=False)
    print(f"\nWrote {pkg_out}")
    print(f"Wrote {cache_out}")
    return pkg_out, cache_out


if __name__ == "__main__":
    main()
