"""Build the unified compound table for the small-molecule chain.

For each compound that's a TF in our coverage set:
  - canonical Broad ID (13-char prefix, e.g. 'BRD-K12345678')
  - pert_iname (human-readable name)
  - SMILES (canonical)
  - InChIKey
  - PubChem CID
  - target gene symbol(s) — pipe-separated
  - MoA classification — for reference (not graded in v1 chain)

Filters:
  - Has SMILES + target + L1000 coverage in >=1 LINCS core-9 line + PRISM viability
  - 966 compounds in ALL 9 LINCS core lines (full-coverage substrate for v1)
"""

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _download import DEFAULT_CACHE

LINCS_CORE = ["A375", "A549", "HEPG2", "HT29", "MCF7", "PC3"]
# Notes on cell-line scope:
#   - VCAP (LINCS core) is not in PRISM viability data — dropped.
#   - HA1E (kidney) and HCC515 (lung) have L1000 data but PRISM has viability
#     for only 55/1042 of our compounds in those lines — dropped.
# This leaves 6 cell lines with full coverage on ~1006 compounds.
CACHE = Path("/tmp/bioreasoning_smallmol_data")


def short_brd(s):
    """Strip salt-batch suffix from a BRD-ID: 'BRD-K12345678-001-02-3' -> 'BRD-K12345678'."""
    if not isinstance(s, str):
        return None
    s = s.replace("BRD:", "")  # PRISM prefix
    return s[:13]


def load_l1000_sig_info(cache_dir: Path) -> pd.DataFrame:
    """Combined sig_info from Phase 1 + Phase 2, compound treatments only."""
    sig1 = pd.read_csv(cache_dir / "sig_info_phase1.txt.gz", sep="\t", low_memory=False)
    sig2 = pd.read_csv(cache_dir / "sig_info_phase2.txt.gz", sep="\t", low_memory=False)
    sig1["phase"] = 1
    sig2["phase"] = 2
    # Align columns; some phase-specific fields default to None
    for c in ["pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]:
        if c not in sig2.columns:
            sig2[c] = None
    sig = pd.concat([sig1, sig2], ignore_index=True)
    sig = sig[sig["pert_type"] == "trt_cp"].copy()
    return sig


def main(cache_dir: Path = CACHE) -> Path:
    print("Building unified compound table...")

    # Drug Repurposing Hub
    drugs = pd.read_csv(cache_dir / "repurposing_drugs.txt", sep="\t", comment="!", dtype=str)
    samples = pd.read_csv(cache_dir / "repurposing_samples.txt", sep="\t", comment="!", dtype=str)
    samples["broad_short"] = samples["broad_id"].apply(short_brd)

    # Join samples (BRD-ID → SMILES + InChIKey) with drugs (pert_iname → target + MoA)
    cmp = samples.merge(drugs[["pert_iname", "moa", "target"]], on="pert_iname", how="left")
    # Drop multi-vendor duplicates: keep first per broad_short
    cmp = cmp.drop_duplicates("broad_short").reset_index(drop=True)
    cmp = cmp.dropna(subset=["broad_short", "smiles", "target"])

    print(f"  Repurposing Hub compounds with SMILES + target: {len(cmp)}")

    # L1000 coverage per LINCS core 8 (VCAP dropped due to PRISM coverage gap)
    sig = load_l1000_sig_info(cache_dir)
    cov = {}
    for line in LINCS_CORE:
        cov[line] = set(sig[sig["cell_id"] == line]["pert_id"].unique())

    # Compounds covered in all 8 lines
    in_all_lines = set.intersection(*cov.values())
    print(f"  L1000 compounds in all {len(LINCS_CORE)} core lines: {len(in_all_lines)}")

    # PRISM viability coverage
    prism = pd.read_csv(cache_dir / "PRISM_Extended_Primary_Compound_List.csv")
    prism["broad_short"] = prism["IDs"].apply(short_brd)
    prism_brds = set(prism["broad_short"].dropna().unique())
    print(f"  PRISM compounds: {len(prism_brds)}")

    # Final intersection: SMILES+target compounds × in-all-N L1000 × PRISM
    full_cov_brds = set(cmp["broad_short"]) & in_all_lines & prism_brds
    print(f"  Full-coverage compounds (SMILES + target + all-{len(LINCS_CORE)} L1000 + PRISM): {len(full_cov_brds)}")

    # Subset compound table to full-coverage set
    cmp_full = cmp[cmp["broad_short"].isin(full_cov_brds)].copy()
    cmp_full = cmp_full[[
        "broad_short", "pert_iname", "smiles", "InChIKey", "pubchem_cid",
        "target", "moa",
    ]].rename(columns={"InChIKey": "inchi_key"})

    out_path = cache_dir / "compound_table.parquet"
    cmp_full.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}: {len(cmp_full)} compounds")
    return out_path


if __name__ == "__main__":
    main()
