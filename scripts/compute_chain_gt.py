"""Compute chain ground-truth for each (compound × cell_line) example — v0.3.

Per row produces:
  Upstream (causal reasoning):
    - target: gene symbols from Drug Repurposing Hub
    - moa: MoA string from Drug Repurposing Hub
    - pathways_signed: top-K Hallmark pathways with up/down direction
                      (signed by mean MODZ z-score of pathway members)

  Downstream phenotypes (each example uses one of these as its prediction target):
    - cell_cycle: {arrest, no_effect, proliferation} from mean(E2F + G2M + MYC) signed
    - stress: {none, apoptosis, UPR, DNA_damage} from argmax of stress pathway signed
    - magnitude: {inert, moderate, strong} from ‖z‖₂ / sqrt(N)
    - viability_lfc: continuous PRISM LFC
    - PRISM assay provenance for the averaged viability target

Output: chain_gt.parquet
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

CACHE = Path("/tmp/bioreasoning_smallmol_data")

# Pathway enrichment
TOP_K_PATHWAYS = 5
GENE_SET_MIN_OVERLAP = 3  # need this many pathway-member genes present in L1000 universe

# Cell-cycle pathway set
CYCLE_SETS = [
    "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1", "HALLMARK_MYC_TARGETS_V2",
]
# Cell-cycle thresholds (clinical: arrest vs proliferation vs no effect)
CYCLE_ARREST_MAX = -0.29
CYCLE_PROLIFERATION_MIN = 0.10

# Stress mechanism pathways
STRESS_SETS = {
    "apoptosis": "HALLMARK_APOPTOSIS",
    "UPR": "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "DNA_damage": "HALLMARK_DNA_REPAIR",
}
STRESS_THRESHOLD = 0.05  # below this, label as "none"

# Transcriptional magnitude buckets (on ‖z‖₂/√N)
MAGNITUDE_INERT_MAX = 0.60
MAGNITUDE_MODERATE_MAX = 0.85

# PRISM Repurposing viability assay endpoint. The 24Q2 release documents a
# 5-day endpoint for the PRISM assay; the compound list carries per-row dose.
PRISM_DURATION_H = 120.0
PRISM_AGGREGATION = "mean_over_full_broad_ids"


# LINCS short code -> DepMap ACH ID for the 6 viable cell lines
ACH_MAP = {
    "A375": "ACH-000219",
    "A549": "ACH-000681",
    "HEPG2": "ACH-000739",
    "HT29": "ACH-000552",
    "MCF7": "ACH-000019",
    "PC3": "ACH-000090",
}


def load_hallmark(path: Path) -> dict[str, set[str]]:
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = set(parts[2:])
    return sets


def signed_pathway_scores(z_series: pd.Series, gene_sets: dict[str, set[str]]) -> dict[str, float]:
    """Per-pathway signed score = mean z over pathway members present in universe."""
    out = {}
    z_clean = z_series.dropna()
    z_index = set(z_clean.index)
    for name, gs in gene_sets.items():
        present = [g for g in gs if g in z_index]
        if len(present) < GENE_SET_MIN_OVERLAP:
            continue
        vals = z_clean.loc[present]
        out[name] = float(vals.mean())
    return out


def top_signed_pathways(scores: dict[str, float], k: int = TOP_K_PATHWAYS) -> list[tuple[str, str]]:
    """Return top-k pathways by |score| with direction label.

    Returns: [(pathway_name, "up" | "down"), ...]
    """
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [(name, "up" if score > 0 else "down") for name, score in ranked]


def cell_cycle_bucket(scores: dict[str, float]) -> str | None:
    vals = [scores[s] for s in CYCLE_SETS if s in scores]
    if not vals:
        return None
    mean_cycle = float(np.mean(vals))
    if mean_cycle <= CYCLE_ARREST_MAX:
        return "arrest"
    if mean_cycle >= CYCLE_PROLIFERATION_MIN:
        return "proliferation"
    return "no_effect"


def stress_bucket(scores: dict[str, float]) -> str | None:
    """Return name of the most-enriched stress pathway, or 'none' if below threshold."""
    avail = {label: scores.get(setname) for label, setname in STRESS_SETS.items()
             if setname in scores}
    if not avail:
        return None
    top_label, top_score = max(avail.items(), key=lambda kv: kv[1])
    if top_score < STRESS_THRESHOLD:
        return "none"
    return top_label


def magnitude_bucket(magnitude: float) -> str:
    if magnitude < MAGNITUDE_INERT_MAX:
        return "inert"
    if magnitude < MAGNITUDE_MODERATE_MAX:
        return "moderate"
    return "strong"


def compute_magnitude(z_series: pd.Series) -> float:
    z = z_series.dropna()
    if z.empty:
        return float("nan")
    return float(np.sqrt((z**2).sum() / len(z)))


def short_brd(s) -> str | None:
    """Strip salt-batch suffix from a BRD-ID."""
    if not isinstance(s, str):
        return None
    s = s.replace("BRD:", "")
    if " - " in s:
        s = s.split(" - ", 1)[0].strip()
    return s[:13]


def full_brd(s) -> str | None:
    """Normalize a full Broad sample ID, preserving suffix when present."""
    if not isinstance(s, str):
        return None
    s = s.replace("BRD:", "").strip()
    if " - " in s:
        s = s.split(" - ", 1)[0].strip()
    return s


def _unique_sorted(values) -> list:
    return sorted(v for v in pd.Series(values).dropna().unique().tolist())


def load_prism_viability(cache_dir: Path) -> dict[tuple[str, str], dict]:
    """Return PRISM viability/provenance keyed by (broad_short, LINCS_cell_code).

    PRISM may contain multiple full Broad sample IDs for the same 13-character
    compound ID. The model cannot observe physical sample identity, so the env
    keeps the existing molecule-level target: mean LFC over full IDs. We still
    carry the exact contributing PRISM dose/screen provenance for prompts/audits.
    """
    path = cache_dir / "PRISM_Extended_Primary_Data_Matrix.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"Unnamed: 0": "compound_id"})
    df["broad_full"] = df["compound_id"].apply(full_brd)
    df["broad_short"] = df["compound_id"].apply(short_brd)

    compound_list_path = cache_dir / "PRISM_Extended_Primary_Compound_List.csv"
    meta = pd.read_csv(compound_list_path, low_memory=False)
    meta["broad_full"] = meta["IDs"].apply(full_brd)
    meta_rows = df[["broad_full", "broad_short"]].merge(
        meta[["broad_full", "screen", "dose"]],
        on="broad_full",
        how="left",
    )
    provenance = {}
    for brd, group in meta_rows.groupby("broad_short"):
        doses = _unique_sorted(group["dose"])
        screens = _unique_sorted(group["screen"])
        provenance[brd] = {
            "prism_dose_um": float(doses[0]) if len(doses) == 1 else None,
            "prism_doses_um": [float(d) for d in doses],
            "prism_duration_h": PRISM_DURATION_H,
            "prism_screens": screens,
            "prism_n_full_ids": int(group["broad_full"].nunique()),
            "prism_aggregation": PRISM_AGGREGATION,
        }

    out = {}
    for line_code, ach_id in ACH_MAP.items():
        if ach_id not in df.columns:
            continue
        sub = df[["broad_short", ach_id]].dropna()
        agg = sub.groupby("broad_short")[ach_id].mean()
        for brd, lfc in agg.items():
            if pd.isna(lfc):
                continue
            out[(brd, line_code)] = {
                "viability_lfc": float(lfc),
                **provenance.get(brd, {}),
            }
    print(f"  PRISM lookup: {len(out)} (compound, line) entries")
    return out


def main(cache_dir: Path = CACHE) -> Path:
    print("Computing chain ground-truth (v0.3)...")

    cmp = pd.read_parquet(cache_dir / "compound_table.parquet")
    l1000 = pd.read_parquet(cache_dir / "l1000_subset.parquet")
    print(f"  Compounds: {len(cmp)}, L1000 sigs: {len(l1000)}")

    prism_lookup = load_prism_viability(cache_dir)
    gene_sets = load_hallmark(cache_dir / "h.all.v2024.1.Hs.symbols.gmt")
    print(f"  Hallmark sets: {len(gene_sets)}")

    gene_cols = [c for c in l1000.columns if c not in (
        "sig_id", "pert_id", "cell_id", "pert_iname", "dose_um", "time_h", "phase"
    )]
    print(f"  Gene universe (L1000 columns): {len(gene_cols)}")

    rows = []
    for i, r in l1000.iterrows():
        if i % 500 == 0:
            print(f"  {i}/{len(l1000)}", flush=True)

        compound = r["pert_id"]
        cell_line = r["cell_id"]

        cmp_row = cmp[cmp["broad_short"] == compound]
        if cmp_row.empty:
            continue
        target_str = cmp_row.iloc[0]["target"]
        targets = [t.strip().upper() for t in str(target_str).split("|") if t.strip()] if pd.notna(target_str) else []
        moa = cmp_row.iloc[0].get("moa")
        if pd.isna(moa) or not isinstance(moa, str) or not moa.strip():
            moa = None
        else:
            moa = moa.strip()
        smiles = cmp_row.iloc[0]["smiles"]

        # L1000 z-scores
        z = pd.to_numeric(pd.Series(r[gene_cols].values, index=gene_cols), errors="coerce")

        # Signed pathway enrichment (all Hallmark sets)
        sc = signed_pathway_scores(z, gene_sets)

        # Top pathways with direction
        pathways_signed = top_signed_pathways(sc)

        # Cell-cycle, stress, magnitude, viability buckets
        cycle = cell_cycle_bucket(sc)
        stress = stress_bucket(sc)
        mag = compute_magnitude(z)
        mag_bucket = magnitude_bucket(mag) if not np.isnan(mag) else None

        prism = prism_lookup.get((compound, cell_line), {})
        lfc = prism.get("viability_lfc")

        rows.append({
            "compound": compound,
            "cell_line": cell_line,
            "smiles": smiles,
            "target": targets,
            "moa": moa,
            "pathways_signed": pathways_signed,
            "cell_cycle": cycle,
            "stress": stress,
            "magnitude": mag_bucket,
            "magnitude_raw": float(mag) if not np.isnan(mag) else None,
            "viability_lfc": lfc,
            "prism_dose_um": prism.get("prism_dose_um"),
            "prism_doses_um": prism.get("prism_doses_um"),
            "prism_duration_h": prism.get("prism_duration_h"),
            "prism_screens": prism.get("prism_screens"),
            "prism_n_full_ids": prism.get("prism_n_full_ids"),
            "prism_aggregation": prism.get("prism_aggregation"),
            "dose_um": r["dose_um"],
            "time_h": r["time_h"],
        })

    df = pd.DataFrame(rows)
    out_path = cache_dir / "chain_gt.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}: {df.shape}")

    # Coverage stats
    for col, label in [
        ("target", "non-empty target"),
        ("moa", "non-empty moa"),
        ("pathways_signed", "non-empty pathways_signed"),
        ("cell_cycle", "non-null cell_cycle"),
        ("stress", "non-null stress"),
        ("magnitude", "non-null magnitude"),
        ("viability_lfc", "non-null viability"),
    ]:
        if col in ("target", "pathways_signed"):
            non_null = df[col].apply(lambda x: bool(x) and len(x) > 0).sum()
        else:
            non_null = df[col].notna().sum()
        print(f"  {label}: {non_null}/{len(df)}")

    print("\nCell-cycle distribution:")
    print(df["cell_cycle"].value_counts(dropna=False).to_string())
    print("\nStress distribution:")
    print(df["stress"].value_counts(dropna=False).to_string())
    print("\nMagnitude distribution:")
    print(df["magnitude"].value_counts(dropna=False).to_string())
    print(f"\nViability LFC: mean {df['viability_lfc'].mean():.3f}, std {df['viability_lfc'].std():.3f}")

    return out_path


if __name__ == "__main__":
    main()
