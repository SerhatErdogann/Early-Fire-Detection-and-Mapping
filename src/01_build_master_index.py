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
    ap.add_argument(
        "--vegetation",
        action="store_true",
        help="Include vegetation parser stub (reserved; no-op unless vegetation parser is implemented).",
    )
    ap.add_argument(
        "--max-cart-samples",
        type=int,
        default=500,
        help="Maximum number of CART paired (color + thermal16) no_fire samples to add (deterministic).",
    )
    ap.add_argument(
        "--cart_in_eval",
        choices=["none", "val", "test", "both"],
        default="none",
        help=(
            "CART placement policy. Default: 'none' = train-only (recommended). "
            "'val'/'test' force CART rows into a single eval split; 'both' keeps the stratified assignment."
        ),
    )
    ap.add_argument(
        "--cart-root",
        default=None,
        help=(
            "CART directory (parent of color/ + thermal16/). "
            "Ignored if FLAME_CART_ROOT is set. "
            "Fallback if neither is set: DATA_ROOT/cart (typically data/cart)."
        ),
    )
    ap.add_argument(
        "--binary-root",
        default=None,
        help=(
            "Standalone multimodal BINARY dataset root with train/, val/, test/ subfolders. "
            "Overrides FLAME_BINARY_ROOT; default if neither: DATA_ROOT/binary (see config)."
        ),
    )
    args = ap.parse_args()
    roots = [Path(p) for p in CUSTOM_DATA_SCAN_ROOTS if p] + [Path(p) for p in args.scan]
    build_master_index(
        scan_custom_roots=roots or None,
        include_vegetation_stub=args.vegetation,
        max_cart_samples=int(args.max_cart_samples),
        cart_in_eval=str(args.cart_in_eval),
        cart_root=args.cart_root,
        binary_root=args.binary_root,
    )


if __name__ == "__main__":
    main()
