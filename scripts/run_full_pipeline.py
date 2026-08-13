"""Run the full chain-substrate build pipeline end-to-end.

Order:
  1. fetch_smallmol.py   (idempotent; skipped if files cached)
  2. build_compound_table.py
  3. filter_l1000.py     (slow: GCTX decompression + cmapPy parse)
  4. compute_chain_gt.py
  5. build_examples.py

Usage:
    uv run python scripts/run_full_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fetch_smallmol
import build_compound_table
import filter_l1000
import compute_chain_gt
import build_examples


def main():
    print("\n=== 1/5 Fetch source files ===")
    fetch_smallmol.main()

    print("\n=== 2/5 Build compound table ===")
    build_compound_table.main()

    print("\n=== 3/5 Filter L1000 ===")
    filter_l1000.main()

    print("\n=== 4/5 Compute chain GT ===")
    compute_chain_gt.main()

    print("\n=== 5/5 Build examples ===")
    build_examples.main()

    print("\nPipeline complete. Next steps:")
    print("  uv pip install -e .  (pick up the data parquet)")
    print("  uv run python scripts/smoke.py")
    print("  prime env push --path . -v PRIVATE")


if __name__ == "__main__":
    main()
