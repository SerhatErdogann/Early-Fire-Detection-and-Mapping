# Segmentation training entry (requires labeled masks in master index).
# Run: python src/03_train_seg.py --help
# Full loop: implement src/training/train_seg.py when masks are available.
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description="Segmentation train scaffold (U-Net baseline)")
    ap.add_argument("--master", default="data/master_index.parquet", help="Master index with path_mask filled")
    args = ap.parse_args()
    p = Path(args.master)
    if not p.exists():
        raise SystemExit(f"Master index not found: {p}. Run 01_build_master_index and add masks.")
    import pandas as pd

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    n_mask = int(df.get("path_mask", pd.Series([""])).astype(str).str.len().gt(0).sum())
    print(f"[03_train_seg] rows={len(df)} with non-empty path_mask ~ {n_mask}")
    print("Implement src/training/train_seg.py (BCE+Dice, fusion 4-ch input) when masks are ready.")


if __name__ == "__main__":
    main()
