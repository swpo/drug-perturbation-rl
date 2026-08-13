"""Filter L1000 Level 5 GCTX files to a usable subset.

Strategy:
  1. Load sig_info; identify the canonical (compound × line × condition)
     signatures we want — standard Touchstone protocol: 10 µM, 24h. For
     compounds without 10µM/24h, fall back to closest condition.
  2. Use cmapPy to read only those signature columns from the big GCTX files
     (avoids loading 25 GB into memory).
  3. Save filtered matrix as parquet: rows = compound × cell_line × phase,
     cols = gene symbols.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

CACHE = Path("/tmp/bioreasoning_smallmol_data")
LINCS_CORE = ["A375", "A549", "HEPG2", "HT29", "MCF7", "PC3"]
TARGET_DOSE_UM = 10.0
TARGET_TIME_H = 24


def pick_canonical_sigs(sig_info: pd.DataFrame,
                        compound_brds: set,
                        cell_lines: list[str]) -> pd.DataFrame:
    """For each (compound, cell_line) pair, pick the canonical signature.

    Priority: 10 µM × 24h > 10 µM × 6h > any 24h > any other.
    Returns a DataFrame with one row per (compound, cell_line) pair.
    """
    df = sig_info[
        (sig_info["pert_id"].isin(compound_brds))
        & (sig_info["cell_id"].isin(cell_lines))
    ].copy()
    # Parse pert_idose to numeric µM (some are "10 µM", "0.1 %", "-666", etc.)
    def parse_dose_um(s):
        if not isinstance(s, str):
            return None
        s = s.strip().lower()
        if "µm" in s or "um " in s or s.endswith("um"):
            try:
                return float(s.replace("µm", "").replace("um", "").strip())
            except ValueError:
                return None
        return None
    df["dose_um"] = df["pert_idose"].apply(parse_dose_um)
    # Parse pert_itime to hours
    def parse_time_h(s):
        if not isinstance(s, str):
            return None
        s = s.strip().lower()
        if s.endswith("h"):
            try:
                return float(s.rstrip("h").strip())
            except ValueError:
                return None
        return None
    df["time_h"] = df["pert_itime"].apply(parse_time_h)

    # Priority scoring: lower = better
    def priority(row):
        score = 0
        if not (row["dose_um"] is not None and abs(row["dose_um"] - TARGET_DOSE_UM) < 0.01):
            score += 100
        if not (row["time_h"] is not None and abs(row["time_h"] - TARGET_TIME_H) < 0.1):
            score += 10
        if row["dose_um"] is None:
            score += 1
        return score
    df["priority"] = df.apply(priority, axis=1)
    df = df.sort_values(["pert_id", "cell_id", "priority"])
    # Pick best per (compound, line)
    picked = df.drop_duplicates(["pert_id", "cell_id"], keep="first").reset_index(drop=True)
    return picked


def load_gctx_subset(gctx_path: Path, sig_ids: list[str], gene_info: pd.DataFrame):
    """Load only the requested signatures (columns) from a GCTX file.

    GCTX files from GEO arrive gzipped; we need an uncompressed .gctx for
    cmapPy. If only the .gz is present, decompress to a sibling .gctx.
    """
    if gctx_path.suffix == ".gz":
        unzipped = gctx_path.with_suffix("")
        if not unzipped.exists():
            import gzip, shutil
            print(f"  Decompressing {gctx_path.name} -> {unzipped.name}...", flush=True)
            with gzip.open(gctx_path, "rb") as src, open(unzipped, "wb") as dst:
                shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        gctx_path = unzipped

    from cmapPy.pandasGEXpress import parse
    print(f"  Loading {gctx_path.name} subset ({len(sig_ids)} signatures)...", flush=True)
    obj = parse.parse(str(gctx_path), cid=sig_ids)
    # data_df is genes (rows, entrez IDs as index) × signatures (cols, sig_ids)
    df = obj.data_df
    # Map entrez index to gene symbols
    g_lookup = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_gene_symbol"]))
    df.index = [g_lookup.get(str(i), str(i)) for i in df.index]
    return df  # genes × sig_ids


def main(cache_dir: Path = CACHE) -> Path:
    print("Filtering L1000 to compound × cell_line subset...")

    # Compound table (from build_compound_table.py)
    cmp = pd.read_parquet(cache_dir / "compound_table.parquet")
    compound_brds = set(cmp["broad_short"])
    print(f"  Compounds to filter: {len(compound_brds)}")

    # Combined sig_info
    sig1 = pd.read_csv(cache_dir / "sig_info_phase1.txt.gz", sep="\t", low_memory=False)
    sig2 = pd.read_csv(cache_dir / "sig_info_phase2.txt.gz", sep="\t", low_memory=False)
    sig1["phase"] = 1
    sig2["phase"] = 2
    for c in ["pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]:
        if c not in sig2.columns:
            sig2[c] = None
    sig = pd.concat([sig1, sig2], ignore_index=True)
    sig = sig[sig["pert_type"] == "trt_cp"].copy()

    # Pick canonical sig per (compound, line)
    picked = pick_canonical_sigs(sig, compound_brds, LINCS_CORE)
    print(f"  Picked {len(picked)} (compound × line) signatures")
    print(f"  Phase 1 sigs: {(picked.phase == 1).sum()}; Phase 2 sigs: {(picked.phase == 2).sum()}")

    gene_info = pd.read_csv(cache_dir / "gene_info.txt.gz", sep="\t", low_memory=False)

    # Load GCTX subsets per phase
    p1_ids = picked[picked.phase == 1]["sig_id"].tolist()
    p2_ids = picked[picked.phase == 2]["sig_id"].tolist()

    matrices = []
    if p1_ids:
        df1 = load_gctx_subset(cache_dir / "GSE92742_Level5_phase1.gctx.gz", p1_ids, gene_info)
        matrices.append(df1)
    if p2_ids:
        df2 = load_gctx_subset(cache_dir / "GSE70138_Level5_phase2.gctx.gz", p2_ids, gene_info)
        matrices.append(df2)

    expr = pd.concat(matrices, axis=1)  # genes × signatures
    print(f"  Combined matrix: {expr.shape} (genes × signatures)")

    # Transpose to signatures × genes. The transposed index is the sig_id.
    expr_t = expr.T
    expr_t.index.name = "sig_id"
    expr_t = expr_t.reset_index()
    # Some signatures may have duplicated gene symbols (after entrez->symbol map);
    # drop dup gene cols by keeping first occurrence
    expr_t = expr_t.loc[:, ~expr_t.columns.duplicated()]

    expr_t = picked[["sig_id", "pert_id", "cell_id", "pert_iname", "dose_um", "time_h", "phase"]].merge(
        expr_t, on="sig_id"
    )

    out_path = cache_dir / "l1000_subset.parquet"
    expr_t.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}: {expr_t.shape}")
    return out_path


if __name__ == "__main__":
    main()
