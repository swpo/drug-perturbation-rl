"""Fetch all source files for the small-molecule chain substrate.

Sources:
  - Drug Repurposing Hub drugs + samples (Broad clue.io S3)
  - LINCS L1000 metadata: sig_info (Phase 1 + 2), pert_info, gene_info
  - LINCS L1000 Level 5 signatures (GCTX, ~25 GB combined)
  - PRISM 24Q2 Extended Primary viability matrix + metadata
  - Hallmark gene sets (MSigDB GMT)

Output dir: /tmp/bioreasoning_smallmol_data/
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _download import download

DEFAULT_CACHE = Path("/tmp/bioreasoning_smallmol_data")


URLS = {
    # Drug Repurposing Hub — small
    "repurposing_drugs.txt": (
        "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
        "repurposing_drugs_20200324.txt"
    ),
    "repurposing_samples.txt": (
        "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
        "repurposing_samples_20200324.txt"
    ),
    # L1000 metadata — small
    "sig_info_phase1.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_sig_info.txt.gz"
    ),
    "sig_info_phase2.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/"
        "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
    ),
    "pert_info.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_pert_info.txt.gz"
    ),
    "gene_info.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_gene_info.txt.gz"
    ),
    # L1000 Level 5 signatures — LARGE (~25 GB combined)
    "GSE92742_Level5_phase1.gctx.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
        "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
    ),
    "GSE70138_Level5_phase2.gctx.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/"
        "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
    ),
    # PRISM viability
    "PRISM_Extended_Primary_Data_Matrix.csv": (
        "https://ndownloader.figshare.com/files/46630984"
    ),
    "PRISM_Extended_Primary_Compound_List.csv": (
        "https://ndownloader.figshare.com/files/46630981"
    ),
    "PRISM_Cell_Line_Meta_Data.csv": "https://ndownloader.figshare.com/files/46630978",
    "PRISM_Treatment_Meta_Data.csv": "https://ndownloader.figshare.com/files/46631146",
    # Hallmark gene sets
    "h.all.v2024.1.Hs.symbols.gmt": (
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
        "2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"
    ),
}


def main(cache_dir: Path = DEFAULT_CACHE) -> dict[str, Path]:
    print(f"Fetching small-molecule chain substrate to {cache_dir}...")
    paths = {}
    for fname, url in URLS.items():
        # min_size=10000 catches truncated GCTX downloads; tiny GMT is exempt
        min_size = 1000 if fname.endswith(".gmt") else 10000
        paths[fname] = download(url, cache_dir / fname, min_size=min_size)
    return paths


if __name__ == "__main__":
    main()
