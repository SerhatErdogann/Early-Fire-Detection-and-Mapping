# Unified master index (Parquet + legacy CSV). Run from project root:
#   python src/01_build_master_index.py
# Optional: scan extra roots from config CUSTOM_DATA_SCAN_ROOTS or --scan
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_master_index import build_master_index

try:
    from config import CUSTOM_DATA_SCAN_ROOTS
except ImportError:
    CUSTOM_DATA_SCAN_ROOTS = []


def main():
    ap = argparse.ArgumentParser(description="Build data/master_index.parquet + outputs/flame_index.csv")
    ap.add_argument(
        "--scan",
        action="append",
        default=[],
        help="Extra folder to scan (fire/no_fire heuristics). Can repeat.",
    )
    ap.add_argument("--vegetation", action="store_true", help="Include vegetation parser stub (no-op until data exists)")
    args = ap.parse_args()
    roots = [Path(p) for p in CUSTOM_DATA_SCAN_ROOTS if p] + [Path(p) for p in args.scan]
    build_master_index(scan_custom_roots=roots or None, include_vegetation_stub=args.vegetation)


if __name__ == "__main__":
    main()
