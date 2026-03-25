"""
Training loop: scene-aware split, loaders, checkpointing, optional calibration metrics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .losses import build_loss
from .metrics import (
    eval_probs,
    eval_logits,
    metrics_at_threshold,
    find_best_threshold_f1,
    fit_temperature,
    expected_calibration_error,
    brier_score_binary,
    _best_threshold_mode,
)
from ..data import FlameDataset
from ..data.path_filter import filter_df_existing_paths
from ..data.split import _group_split_three_way, split_train_val_extra
from ..models import make_classifier, get_model_config

try:
    from config import TRAIN_DEFAULT, MODELS_DIR, OUTPUTS_DIR
except ImportError:
    TRAIN_DEFAULT = {}
    MODELS_DIR = Path("models")
    OUTPUTS_DIR = Path("outputs")


def _load_index(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["label"]).copy()
    if "path_th" not in df.columns and "path_thermal" in df.columns:
        df["path_th"] = df["path_thermal"]
    if "label_fire" not in df.columns:
        df["label_fire"] = df["label"].astype(int)
    if "split_group" not in df.columns:
        if "key" in df.columns and "source" in df.columns:
            df["split_group"] = df["source"].astype(str) + "_" + df["key"].astype(str)
        else:
            df["split_group"] = df.index.astype(str)
    return df


def _split_data(
    df: pd.DataFrame,
    extra_test_ratio: float,
    val_split: float,
    flame_test_ratio: float,
    random_state: int = 42,
):
    """
    Scene/group-level split on ``split_group``. ``extra`` (drone no-fire) uses group holdout.
    All non-extra rows (flame3, binary, custom) share one group split so binary is included in train/val/test.
    """
    rng = np.random.default_rng(random_state)
    df = _prepare_df(df)
    df_main, extra_test_df = split_train_val_extra(df, extra_test_ratio=extra_test_ratio, random_state=random_state)
    main_non_extra = df_main[df_main["source"] != "extra"].copy()
    if len(main_non_extra) == 0:
        raise SystemExit("No training rows after removing extra holdout split.")
    tr, va, test_df = _group_split_three_way(main_non_extra, flame_test_ratio, val_split, rng)
    extra_train = df_main[df_main["source"] == "extra"]
    if len(extra_train) > 0:
        tr = pd.concat([tr, extra_train], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return tr, va, test_df, extra_test_df


def _sample_weights(tr: pd.DataFrame, loss_mode: str, hard_paths: set[str] | None) -> np.ndarray | None:
    use_sampler = loss_mode in ("sampler_ce", "sampler_focal") or loss_mode.startswith("sampler_")
    if not use_sampler:
        return None
    counts = tr["label"].value_counts().to_dict()
    n0, n1 = int(counts.get(0, 1)), int(counts.get(1, 1))
    w_per_class = {0: 1.0 / max(1, n0), 1: 1.0 / max(1, n1)}
    sw = tr["label"].map(w_per_class).astype(np.float64)
    if "label_quality" in tr.columns:
        qmap = {"gold": 1.0, "silver": 0.85, "weak": 0.65}
        sw = sw * tr["label_quality"].map(qmap).fillna(0.75).astype(np.float64)
    if hard_paths:
        hp = tr["path_rgb"].astype(str).isin(hard_paths)
        sw = sw * (1.0 + hp.astype(np.float64))
    return sw.values.astype(np.float32)


def train_one_run(
    csv_path,
    mode="fusion",
    epochs=20,
    bs=16,
    lr=1e-4,
    size=384,
    out_ckpt=None,
    loss_mode="sampler_focal",
    extra_test_ratio=0.2,
    val_split=0.2,
    flame_test_ratio=0.1,
    patience=4,
    backbone="resnet18",
    use_amp=True,
    focal_gamma=2.0,
    scheduler_kind="plateau",
    model_family: str | None = None,
    calibrate_report: bool = True,
    hard_negative_csv: str | None = None,
    save_oof_predictions: bool = False,
    loss_name: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()
    df = _load_index(csv_path)
    df = _prepare_df(df)

    mf = (model_family or "early_fusion").lower()
    if mf == "rgb_baseline":
        mode = "rgb"
    elif mf == "thermal_baseline":
        mode = "thermal"
    elif mf in ("early_fusion", "dual_branch_fusion"):
        mode = "fusion"

    df, drop_tr = filter_df_existing_paths(df, mode=mode)
    if drop_tr:
        print(f"[train] Dropped {drop_tr} rows (missing files on disk for mode={mode!r})")

    tr, va, test_df, extra_test_df = _split_data(df, extra_test_ratio, val_split, flame_test_ratio)
    print(f"\n[{mode}/{mf}] train={len(tr)} | val={len(va)} | test={len(test_df)} | extra_test={len(extra_test_df)}")

    hard_paths: set[str] | None = None
    if hard_negative_csv and Path(hard_negative_csv).exists():
        hdf = pd.read_csv(hard_negative_csv)
        col = "path_rgb" if "path_rgb" in hdf.columns else hdf.columns[0]
        hard_paths = set(hdf[col].astype(str).tolist())
        print(f"[train] hard_negative_csv: {len(hard_paths)} paths")

    in_ch, _ = get_model_config(mode)
    model = make_classifier(mf, backbone, mode, num_classes=2, pretrained=True).to(device)

    sw_arr = _sample_weights(tr, loss_mode, hard_paths)
    sampler = (
        WeightedRandomSampler(
            weights=torch.tensor(sw_arr, dtype=torch.float32),
            num_samples=len(tr),
            replacement=True,
        )
        if sw_arr is not None
        else None
    )

    train_loader = DataLoader(
        FlameDataset(tr, mode=mode, size=size, train=True),
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        FlameDataset(va, mode=mode, size=size, train=False),
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        FlameDataset(test_df, mode=mode, size=size, train=False),
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )
    extra_test_loader = None
    if len(extra_test_df) > 0:
        extra_test_loader = DataLoader(
            FlameDataset(extra_test_df, mode=mode, size=size, train=False),
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=use_cuda,
        )

    counts = tr["label"].value_counts().to_dict()
    n0, n1 = int(counts.get(0, 1)), int(counts.get(1, 1))
    w0 = (n0 + n1) / (2 * n0)
    w1 = (n0 + n1) / (2 * n1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    class_counts = torch.tensor([float(n0), float(n1)], dtype=torch.float32).to(device)
    ln = loss_name or (
        "ce"
        if loss_mode == "sampler_ce"
        else ("cb_focal" if loss_mode == "class_balanced_focal" else "focal")
    )
    loss_fn = build_loss(ln, class_weights, device, focal_gamma=focal_gamma, class_counts=class_counts)
    loss_fn = loss_fn.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if scheduler_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if (use_cuda and use_amp) else None

    out_ckpt = out_ckpt or str(MODELS_DIR / f"{mode}.pt")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    best_val_ap = -1.0
    patience_counter = 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{mode} ep{ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        vy, vp = eval_probs(model, val_loader, device)
        best_thr = find_best_threshold_f1(vy, vp)
        vm = metrics_at_threshold(vy, vp, best_thr)
        ece = expected_calibration_error(vy, vp)
        brier = brier_score_binary(vy, vp)
        print(
            f"Val acc={vm['acc']:.3f} auc={vm['auc']:.3f} ap={vm['ap']:.3f} thr={best_thr:.2f} "
            f"P={vm['precision']:.3f} R={vm['recall']:.3f} F1={vm['f1']:.3f} ECE={ece:.3f} Brier={brier:.3f}"
        )
        print("Val CM [[TN FP],[FN TP]]:\n", vm["cm"])

        ty, tp = eval_probs(model, test_loader, device)
        tm = metrics_at_threshold(ty, tp, best_thr)
        print(f"Test acc={tm['acc']:.3f} auc={tm['auc']:.3f} ap={tm['ap']:.3f}")

        if extra_test_loader is not None:
            ey, ep_probs = eval_probs(model, extra_test_loader, device)
            pred = (ep_probs >= best_thr).astype(np.int64)
            n_fp = int((pred == 1).sum())
            n_tn = int((pred == 0).sum())
            fp_rate = n_fp / max(1, len(ey))
            print(f"Extra test (drone no-fire) n={len(ey)} FP={n_fp} TN={n_tn} FP_rate={fp_rate:.3f}")

        if scheduler_kind == "plateau":
            sched.step(vm["ap"] if vm["ap"] == vm["ap"] else 0.0)
        else:
            sched.step()

        if vm["ap"] == vm["ap"] and float(vm["ap"]) > best_val_ap:
            best_val_ap = float(vm["ap"])
            patience_counter = 0
            vy_log, val_logits = eval_logits(model, val_loader, device)
            T = fit_temperature(vy_log, val_logits)
            thr_alarm = _best_threshold_mode(vy, vp, "alarm")
            thr_review = _best_threshold_mode(vy, vp, "review")
            torch.save(
                {
                    "mode": mode,
                    "model_family": mf,
                    "in_ch": in_ch,
                    "backbone": backbone,
                    "state": model.state_dict(),
                    "threshold": float(best_thr),
                    "threshold_alarm": float(thr_alarm),
                    "threshold_review": float(thr_review),
                    "val_ap": float(vm["ap"]),
                    "temperature": float(T),
                },
                out_ckpt,
            )
            print(f"Saved (best val AP, T={T:.3f}): {out_ckpt}")
            extra_info = {}
            if extra_test_loader is not None:
                extra_info = {
                    "extra_test_n": len(ey),
                    "extra_test_fp": n_fp,
                    "extra_test_tn": n_tn,
                    "extra_test_fp_rate": fp_rate,
                }
            metrics = {
                "mode": mode,
                "model_family": mf,
                "epoch": ep,
                "threshold": float(best_thr),
                "threshold_alarm": float(thr_alarm),
                "threshold_review": float(thr_review),
                "temperature": float(T),
                "ece_val": float(ece),
                "brier_val": float(brier),
                "val": {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in vm.items()},
                "test": {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in tm.items()},
                **extra_info,
            }
            metrics_path = Path(OUTPUTS_DIR) / f"metrics_{mode}_{mf}.json"
            if mf == "early_fusion" and mode == "fusion":
                metrics_path = Path(OUTPUTS_DIR) / f"metrics_{mode}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            if save_oof_predictions:
                oof = pd.DataFrame(
                    {
                        "y_true": vy,
                        "prob_fire": vp,
                        "split": "val",
                    }
                )
                oof_path = Path(OUTPUTS_DIR) / f"oof_{mode}_{mf}.csv"
                oof.to_csv(oof_path, index=False)
                print("OOF val predictions:", oof_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop (val AP did not improve for {patience} epochs)")
                break

    if calibrate_report and Path(out_ckpt).exists():
        ck = torch.load(out_ckpt, map_location=device)
        T = float(ck.get("temperature", 1.0))
        print(f"[calibrate] checkpoint temperature={T:.4f} thresholds alarm={ck.get('threshold_alarm')} review={ck.get('threshold_review')}")

    return out_ckpt
