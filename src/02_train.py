# Train RGB / thermal / fusion / dual-branch models. Run from project root: python src/02_train.py
import argparse
from pathlib import Path
import sys

sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from src.training.trainer import train_one_run

try:
    from config import (
        FLAME_INDEX_CSV,
        MASTER_INDEX_PARQUET,
        MODELS_DIR,
        TRAIN_DEFAULT,
        CKPT_DUAL_BRANCH,
        FUSION_TRAIN_DEFAULT,
    )
except ImportError:
    FLAME_INDEX_CSV = "outputs/flame_index.csv"
    MASTER_INDEX_PARQUET = "data/master_index.parquet"
    MODELS_DIR = Path("models")
    TRAIN_DEFAULT = {}
    CKPT_DUAL_BRANCH = Path("models/dual_branch.pt")
    FUSION_TRAIN_DEFAULT = {}


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
    # Sentinels (None) let us apply mode-aware defaults after parsing. For fusion
    # we prefer FUSION_TRAIN_DEFAULT unless the user passed an explicit value.
    ap.add_argument("--loss_mode", choices=["sampler_ce", "focal_shuffle", "sampler_focal", "balanced_sampler"], default=None)
    ap.add_argument(
        "--loss_name",
        default=None,
        help="Loss implementation: focal | ce | weighted_ce | cb_focal | label_smoothing_ce",
    )
    ap.add_argument("--epochs", type=int, default=TRAIN_DEFAULT.get("epochs", 20))
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--extra_test_ratio", type=float, default=TRAIN_DEFAULT.get("extra_test_ratio", 0.2))
    ap.add_argument("--size", type=int, default=None, help="Input size (e.g. 384)")
    ap.add_argument("--bs", type=int, default=TRAIN_DEFAULT.get("batch_size", 16))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone", default=None, help="resnet18 | resnet50 | efficientnet_b0")
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--no_calibrate_report", action="store_true", help="Skip calibration summary printout after training")
    ap.add_argument("--hard_negative_csv", default=None, help="CSV with path_rgb column for hard negatives (upweighted)")
    ap.add_argument("--save_oof_predictions", action="store_true", help="Save validation OOF probs on best epoch")
    ap.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument(
        "--exclude_sources",
        default="",
        help="Comma-separated sources to exclude (ablation), e.g. flame_video_nofire,binary_root",
    )
    ap.add_argument("--max_train_batches", type=int, default=0, help="Debug: cap train batches per epoch (0=no cap)")
    ap.add_argument(
        "--max_val_batches",
        type=int,
        default=-1,
        help="Debug: val batches cap. -1=full val (default), 0=skip val, N>0=cap to N batches",
    )
    ap.add_argument("--num_workers", type=int, default=-1, help="Override DataLoader workers (-1 = default)")
    ap.add_argument("--pin_memory", type=int, default=-1, help="DataLoader pin_memory (0/1). -1=default")
    ap.add_argument("--prefetch_factor", type=int, default=-1, help="DataLoader prefetch_factor. -1=default")
    ap.add_argument("--persistent_workers", type=int, default=-1, help="DataLoader persistent_workers (0/1). -1=default")
    ap.add_argument(
        "--thermal_norm",
        default="",
        help="Thermal normalization: percentile|minmax|uint16_div (empty=default)",
    )
    ap.add_argument(
        "--no_fire_weight",
        type=float,
        default=1.0,
        help="Multiplies per-class loss emphasis for label=0 (no_fire). 1.0=default.",
    )
    ap.add_argument(
        "--fire_weight",
        type=float,
        default=1.0,
        help="Multiplies per-class loss emphasis for label=1 (fire). 1.0=default.",
    )
    ap.add_argument(
        "--flame_video_nofire_weight",
        type=float,
        default=-1.0,
        help="Sampling weight override for source=flame_video_nofire (e.g. 1.8). -1=default",
    )
    ap.add_argument(
        "--inference_threshold",
        type=float,
        default=-1.0,
        help="Override default inference threshold saved in checkpoint (e.g. 0.55). -1=auto",
    )
    ap.add_argument(
        "--selection_metric",
        choices=["f1_balacc", "realistic"],
        default="f1_balacc",
        help="Checkpoint selection: legacy 0.5*f1+0.5*bal_acc vs realistic f1+bal_acc-0.5*fpr (val @ op thr).",
    )
    ap.add_argument(
        "--source_weights",
        default="",
        help=(
            "Per-source sampling weight overrides as 'k1=v1,k2=v2'. "
            "E.g. 'cart_aux=0.5,flame_video_nofire=1.5'. Empty = use defaults."
        ),
    )
    ap.add_argument(
        "--modal_dropout_p",
        type=float,
        default=0.0,
        help=(
            "Fusion-only modal dropout probability per batch. With prob p we zero "
            "either RGB (channels 0:3) or thermal (channel 3). 0.10 recommended."
        ),
    )
    args = ap.parse_args()

    default_index = MASTER_INDEX_PARQUET if Path(MASTER_INDEX_PARQUET).exists() else FLAME_INDEX_CSV
    csv_path = args.csv or str(default_index)
    if not Path(csv_path).exists():
        raise SystemExit(
            f"Index not found: {csv_path}. Run python src/01_build_master_index.py first."
        )

    modes = ["rgb", "thermal", "fusion"] if args.mode == "all" else [args.mode]
    ckpts = {
        "rgb": str(MODELS_DIR / "rgb.pt"),
        "thermal": str(MODELS_DIR / "thermal.pt"),
        "fusion": str(MODELS_DIR / "fusion.pt"),
    }

    # Fusion is now dual-branch by default (separate RGB / thermal encoders,
    # features concat -> classifier). Override per --model_family if the user
    # passes it explicitly.
    fusion_family_default = str(FUSION_TRAIN_DEFAULT.get("model_family", "dual_branch_fusion"))

    def family_for_mode(m: str) -> str:
        if args.model_family:
            if args.model_family == "dual_branch_fusion" and m != "fusion":
                return "rgb_baseline" if m == "rgb" else "thermal_baseline"
            return args.model_family
        if m == "rgb":
            return "rgb_baseline"
        if m == "thermal":
            return "thermal_baseline"
        return fusion_family_default

    def resolve_defaults(m: str, mf: str) -> dict:
        """Return the effective training hyperparameters for this run, applying
        fusion-specific defaults only when the CLI flag was left as a sentinel.
        """
        is_fusion = (m == "fusion")
        fd = FUSION_TRAIN_DEFAULT if is_fusion else {}
        # Backbone: dual_branch_fusion prefers resnet50 (efficientnet_b0 works
        # too but resnet50 gives a closer apples-to-apples comparison against
        # early fusion). Early fusion (single 4-channel encoder) defaults to
        # efficientnet_b0 for a stronger baseline.
        if args.backbone is not None:
            backbone = str(args.backbone)
        elif is_fusion and mf == "dual_branch_fusion":
            backbone = str(fd.get("dual_branch_backbone", "resnet50"))
        elif is_fusion:
            backbone = str(fd.get("backbone", "efficientnet_b0"))
        else:
            backbone = "resnet18"

        return {
            "loss_mode": args.loss_mode or (fd.get("loss_mode") if is_fusion else "sampler_focal"),
            "loss_name": args.loss_name if args.loss_name is not None else (fd.get("loss_name") if is_fusion else None),
            "lr": float(args.lr) if args.lr is not None else float(fd.get("lr", TRAIN_DEFAULT.get("lr", 1e-4))),
            "patience": int(args.patience) if args.patience is not None else int(fd.get("patience", TRAIN_DEFAULT.get("patience", 4))),
            "size": int(args.size) if args.size is not None else int(fd.get("size", TRAIN_DEFAULT.get("size", 384))),
            "backbone": backbone,
            "focal_gamma": float(fd.get("focal_gamma", TRAIN_DEFAULT.get("focal_gamma", 2.0))) if is_fusion else float(TRAIN_DEFAULT.get("focal_gamma", 2.0)),
        }

    for m in modes:
        mf = family_for_mode(m)
        resolved = resolve_defaults(m, mf)
        out = ckpts[m]
        if mf == "dual_branch_fusion":
            out = str(CKPT_DUAL_BRANCH)
        print(
            f"[02_train] mode={m} family={mf} backbone={resolved['backbone']} "
            f"size={resolved['size']} lr={resolved['lr']} patience={resolved['patience']} "
            f"loss_mode={resolved['loss_mode']} loss_name={resolved['loss_name']} focal_gamma={resolved['focal_gamma']}"
        )
        train_one_run(
            csv_path,
            mode=m,
            epochs=args.epochs,
            bs=args.bs,
            lr=resolved["lr"],
            size=resolved["size"],
            out_ckpt=out,
            loss_mode=resolved["loss_mode"],
            loss_name=resolved["loss_name"],
            extra_test_ratio=args.extra_test_ratio,
            val_split=TRAIN_DEFAULT.get("val_split", 0.2),
            flame_test_ratio=TRAIN_DEFAULT.get("flame_test_ratio", 0.1),
            patience=resolved["patience"],
            backbone=resolved["backbone"],
            use_amp=not args.no_amp,
            focal_gamma=float(resolved["focal_gamma"]),
            scheduler_kind=TRAIN_DEFAULT.get("scheduler", "plateau"),
            model_family=mf,
            calibrate_report=not args.no_calibrate_report,
            hard_negative_csv=args.hard_negative_csv,
            save_oof_predictions=args.save_oof_predictions,
            grad_accum_steps=args.grad_accum_steps,
            exclude_sources=[s.strip() for s in str(args.exclude_sources).split(",") if s.strip()],
            max_train_batches=(None if int(args.max_train_batches) <= 0 else int(args.max_train_batches)),
            max_val_batches=(None if int(args.max_val_batches) < 0 else int(args.max_val_batches)),
            num_workers=(None if int(args.num_workers) < 0 else int(args.num_workers)),
            pin_memory=(None if int(args.pin_memory) < 0 else bool(int(args.pin_memory))),
            prefetch_factor=(None if int(args.prefetch_factor) < 0 else int(args.prefetch_factor)),
            persistent_workers=(None if int(args.persistent_workers) < 0 else bool(int(args.persistent_workers))),
            thermal_norm=(None if not str(args.thermal_norm).strip() else str(args.thermal_norm).strip()),
            flame_video_nofire_weight=(None if float(args.flame_video_nofire_weight) < 0 else float(args.flame_video_nofire_weight)),
            inference_threshold=(None if float(args.inference_threshold) < 0 else float(args.inference_threshold)),
            no_fire_weight=float(args.no_fire_weight),
            fire_weight=float(args.fire_weight),
            selection_metric=str(args.selection_metric),
            source_weights=str(args.source_weights),
            modal_dropout_p=float(args.modal_dropout_p),
        )


if __name__ == "__main__":
    main()
