# Train RGB / thermal / fusion / dual-branch models. Run from project root: python src/02_train.py
import argparse
from pathlib import Path
import sys

sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from src.training.trainer import train_one_run

try:
    from config import FLAME_INDEX_CSV, MASTER_INDEX_PARQUET, MODELS_DIR, TRAIN_DEFAULT, CKPT_DUAL_BRANCH
except ImportError:
    FLAME_INDEX_CSV = "outputs/flame_index.csv"
    MASTER_INDEX_PARQUET = "data/master_index.parquet"
    MODELS_DIR = Path("models")
    TRAIN_DEFAULT = {}
    CKPT_DUAL_BRANCH = Path("models/dual_branch.pt")


def main():
    ap = argparse.ArgumentParser(description="Train fire classification (RGB / thermal / fusion / dual-branch)")
    ap.add_argument("--csv", default=None, help="Index CSV or Parquet (default: master parquet if exists else flame CSV)")
    ap.add_argument("--mode", choices=["rgb", "thermal", "fusion", "all"], default="all")
    ap.add_argument(
        "--model_family",
        choices=["rgb_baseline", "thermal_baseline", "early_fusion", "dual_branch_fusion"],
        default=None,
        help="If set, overrides per-mode default (use with single --mode)",
    )
    ap.add_argument("--loss_mode", choices=["sampler_ce", "focal_shuffle", "sampler_focal"], default="sampler_focal")
    ap.add_argument(
        "--loss_name",
        default=None,
        help="Loss implementation: focal | ce | weighted_ce | cb_focal | label_smoothing_ce",
    )
    ap.add_argument("--epochs", type=int, default=TRAIN_DEFAULT.get("epochs", 20))
    ap.add_argument("--patience", type=int, default=TRAIN_DEFAULT.get("patience", 4))
    ap.add_argument("--extra_test_ratio", type=float, default=TRAIN_DEFAULT.get("extra_test_ratio", 0.2))
    ap.add_argument("--size", type=int, default=TRAIN_DEFAULT.get("size", 384), help="Input size (e.g. 384)")
    ap.add_argument("--bs", type=int, default=TRAIN_DEFAULT.get("batch_size", 16))
    ap.add_argument("--lr", type=float, default=TRAIN_DEFAULT.get("lr", 1e-4))
    ap.add_argument("--backbone", default="resnet18", help="resnet18 | resnet50 | efficientnet_b0")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--no_calibrate_report", action="store_true", help="Skip calibration summary printout after training")
    ap.add_argument("--hard_negative_csv", default=None, help="CSV with path_rgb column for hard negatives (upweighted)")
    ap.add_argument("--save_oof_predictions", action="store_true", help="Save validation OOF probs on best epoch")
    args = ap.parse_args()

    default_index = MASTER_INDEX_PARQUET if Path(MASTER_INDEX_PARQUET).exists() else FLAME_INDEX_CSV
    csv_path = args.csv or str(default_index)
    if not Path(csv_path).exists():
        raise SystemExit(f"Index not found: {csv_path}. Run python src/01_build_index.py first.")

    modes = ["rgb", "thermal", "fusion"] if args.mode == "all" else [args.mode]
    ckpts = {
        "rgb": str(MODELS_DIR / "rgb.pt"),
        "thermal": str(MODELS_DIR / "thermal.pt"),
        "fusion": str(MODELS_DIR / "fusion.pt"),
    }

    def family_for_mode(m: str) -> str:
        if args.model_family:
            if args.model_family == "dual_branch_fusion" and m != "fusion":
                return "rgb_baseline" if m == "rgb" else "thermal_baseline"
            return args.model_family
        if m == "rgb":
            return "rgb_baseline"
        if m == "thermal":
            return "thermal_baseline"
        return "early_fusion"

    for m in modes:
        mf = family_for_mode(m)
        out = ckpts[m]
        if mf == "dual_branch_fusion":
            out = str(CKPT_DUAL_BRANCH)
        train_one_run(
            csv_path,
            mode=m,
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            size=args.size,
            out_ckpt=out,
            loss_mode=args.loss_mode,
            loss_name=args.loss_name,
            extra_test_ratio=args.extra_test_ratio,
            val_split=TRAIN_DEFAULT.get("val_split", 0.2),
            flame_test_ratio=TRAIN_DEFAULT.get("flame_test_ratio", 0.1),
            patience=args.patience,
            backbone=args.backbone,
            use_amp=not args.no_amp,
            focal_gamma=TRAIN_DEFAULT.get("focal_gamma", 2.0),
            scheduler_kind=TRAIN_DEFAULT.get("scheduler", "plateau"),
            model_family=mf,
            calibrate_report=not args.no_calibrate_report,
            hard_negative_csv=args.hard_negative_csv,
            save_oof_predictions=args.save_oof_predictions,
        )


if __name__ == "__main__":
    main()
