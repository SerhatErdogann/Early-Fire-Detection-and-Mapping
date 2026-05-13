"""
Training loop: scene-aware split, loaders, checkpointing, optional calibration metrics.
"""
from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

from .eval_reporting import (
    recall_fpr_selection_key,
    sanitize_for_json,
    realistic_selection_score,
)
from .losses import build_loss
from .metrics import (
    metrics_at_threshold,
    find_best_threshold_f1,
    fit_temperature,
    expected_calibration_error,
    brier_score_binary,
    _best_threshold_mode,
)
from ..data import FlameDataset
from ..eval.robustness_eval import CORRUPTIONS, eval_logits_corrupted, protocol_corruption
from ..data.path_filter import filter_df_existing_paths
from ..data.split import _group_split_three_way, split_train_val_extra
from ..models import FUSION_DUAL_FAMILIES, make_classifier, get_model_config

try:
    from config import TRAIN_DEFAULT, MODELS_DIR, OUTPUTS_DIR, THRESHOLD_ALARM_MIN
except ImportError:
    TRAIN_DEFAULT = {}
    MODELS_DIR = Path("models")
    OUTPUTS_DIR = Path("outputs")
    THRESHOLD_ALARM_MIN = 0.25

def _fmt_bytes(n: float | int) -> str:
    n = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _ram_snapshot() -> dict:
    # Prefer psutil if available (not required dependency).
    try:  # pragma: no cover
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        return {
            "kind": "psutil",
            "rss_bytes": int(mi.rss),
            "vms_bytes": int(getattr(mi, "vms", 0)),
        }
    except Exception:
        return {"kind": "unknown", "rss_bytes": None, "vms_bytes": None}


def _gpu_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"kind": "no_cuda"}
    try:
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        return {
            "kind": "cuda",
            "device": int(dev),
            "name": str(props.name),
            "total_bytes": int(props.total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
            "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
        }
    except Exception:
        return {"kind": "cuda", "error": "snapshot_failed"}


def _print_first_batch_info(train_loader: DataLoader, device: str, pin_memory: bool) -> None:
    try:
        xb, yb = next(iter(train_loader))
    except Exception as e:
        print(f"[smoke] first batch fetch failed: {type(e).__name__}: {e}")
        return
    print(f"[smoke] first batch cpu shapes: x={tuple(xb.shape)} y={tuple(yb.shape)} x.dtype={xb.dtype} y.dtype={yb.dtype}")
    nb = bool(pin_memory) and device == "cuda"
    try:
        xb2 = xb.to(device, non_blocking=nb)
        yb2 = yb.to(device, non_blocking=nb)
        print(f"[smoke] first batch device={device} shapes: x={tuple(xb2.shape)} y={tuple(yb2.shape)}")
        # free refs ASAP
        del xb2, yb2
        if device == "cuda":
            torch.cuda.synchronize()
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda out of memory" in msg or "out of memory" in msg:
            print(f"[smoke][OOM][CUDA] moving first batch to GPU caused OOM: {e}")
        else:
            print(f"[smoke] moving first batch to device failed: {type(e).__name__}: {e}")
    except MemoryError as e:
        print(f"[smoke][OOM][RAM] moving first batch caused RAM OOM: {e}")


def _probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    lt = np.asarray(logits, dtype=np.float64) / max(1e-6, float(temperature))
    lt = lt - np.max(lt, axis=1, keepdims=True)
    ex = np.exp(lt)
    sm = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)
    return sm[:, 1].astype(np.float32)


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


def _hard_negative_hits(tr: pd.DataFrame, hard_paths: set[str] | None, hard_keys: set[str] | None) -> pd.Series:
    """Boolean mask aligned with ``tr`` rows: training rows matching hard-negative lists."""
    hits = pd.Series(False, index=tr.index)
    if not hard_paths and not hard_keys:
        return hits
    if hard_paths:
        if "path_rgb" in tr.columns:
            hits = hits | tr["path_rgb"].astype(str).isin(hard_paths)
        if "path_th" in tr.columns:
            hits = hits | tr["path_th"].astype(str).isin(hard_paths)
    if hard_keys and "key" in tr.columns:
        hits = hits | tr["key"].astype(str).isin(hard_keys)
    return hits


def _extra_hard_negative_class0_indices(tr: pd.DataFrame, hard_paths: set[str] | None, hard_keys: set[str] | None) -> list[int]:
    """Duplicate row indices (for class-0 pool only) so hard-negative no_fire rows appear more often."""
    if not hard_paths and not hard_keys:
        return []
    hn = _hard_negative_hits(tr, hard_paths, hard_keys)
    lab = tr["label"].astype(int).to_numpy()
    m = hn.to_numpy(dtype=bool) & (lab == 0)
    return np.flatnonzero(m).astype(np.int64).tolist()


def _label_counts_dict(d: pd.DataFrame) -> dict:
    vc = d["label"].value_counts().to_dict() if len(d) and "label" in d.columns else {}
    n = int(len(d))
    n0 = int(vc.get(0, 0))
    n1 = int(vc.get(1, 0))
    out = {
        "n": n,
        "label_0_no_fire": n0,
        "label_1_fire": n1,
    }
    if n > 0:
        out["pct_0_no_fire"] = round(100.0 * n0 / n, 4)
        out["pct_1_fire"] = round(100.0 * n1 / n, 4)
    return out


def _source_counts_breakdown(d: pd.DataFrame) -> dict:
    """Row counts per ``source`` for logging / JSON (empty if no column)."""
    if len(d) == 0 or "source" not in d.columns:
        return {}
    vc = d["source"].astype(str).value_counts()
    return {str(k): int(v) for k, v in sorted(vc.items(), key=lambda kv: (-kv[1], kv[0]))}


def _per_source_label_breakdown(d: pd.DataFrame) -> dict:
    """Per-source no_fire / fire counts (empty if no ``source`` column)."""
    if len(d) == 0 or "source" not in d.columns:
        return {}
    out: dict[str, dict] = {}
    for s in sorted(d["source"].astype(str).unique()):
        sub = d[d["source"].astype(str) == s]
        out[str(s)] = _label_counts_dict(sub)
    return out


class BalancedTwoClassBatchSampler(Sampler[list[int]]):
    """
    Each batch: ``batch_size//2`` indices from class 0 and the rest from class 1
    (deterministic split; odd ``batch_size`` gives one extra minority-class slot).
    Indices are drawn with replacement within an epoch-sized pass (shuffled per epoch).
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        extra_class0_indices: list[int] | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.bs = max(2, int(batch_size))
        self.n0 = max(1, self.bs // 2)
        self.n1 = self.bs - self.n0
        idx0 = np.where(self.labels == 0)[0].astype(np.int64)
        idx1 = np.where(self.labels == 1)[0].astype(np.int64)
        if extra_class0_indices:
            idx0 = np.concatenate([idx0, np.asarray(extra_class0_indices, dtype=np.int64)])
        if len(idx0) == 0 or len(idx1) == 0:
            raise ValueError("balanced_sampler requires both class 0 and class 1 rows in the training set.")
        self._idx0 = idx0
        self._idx1 = idx1

    def __len__(self) -> int:
        return int(max((len(self._idx0) + self.n0 - 1) // self.n0, (len(self._idx1) + self.n1 - 1) // self.n1))

    def __iter__(self):
        rng = np.random.default_rng()
        s0 = rng.permutation(self._idx0)
        s1 = rng.permutation(self._idx1)
        p0 = p1 = 0
        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(self.n0):
                batch.append(int(s0[p0 % len(s0)]))
                p0 += 1
            for _ in range(self.n1):
                batch.append(int(s1[p1 % len(s1)]))
                p1 += 1
            rng.shuffle(batch)  # break fixed [all0][all1] order inside the batch
            yield batch


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

    # If master index already provides an explicit split, respect it.
    if "split" in df.columns:
        sp = df["split"].astype(str).str.lower().str.strip()
        ok = sp.isin(["train", "val", "test"])
        if int(ok.sum()) > 0:
            df2 = df.copy()
            df2["split"] = sp.where(ok, "")
            tr = df2[df2["split"] == "train"].copy()
            va = df2[df2["split"] == "val"].copy()
            te = df2[df2["split"] == "test"].copy()
            extra_test_df = df2.iloc[0:0].copy()
            return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True), extra_test_df

    df_main, extra_test_df = split_train_val_extra(df, extra_test_ratio=extra_test_ratio, random_state=random_state)
    main_non_extra = df_main[df_main["source"] != "extra"].copy()
    if len(main_non_extra) == 0:
        raise SystemExit("No training rows after removing extra holdout split.")
    tr, va, test_df = _group_split_three_way(main_non_extra, flame_test_ratio, val_split, rng)
    extra_train = df_main[df_main["source"] == "extra"]
    if len(extra_train) > 0:
        tr = pd.concat([tr, extra_train], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return tr, va, test_df, extra_test_df


def _parse_source_weights(s: str | None) -> dict[str, float]:
    """Parse a CLI string ``"k1=v1,k2=v2"`` into ``{k1: float(v1), k2: float(v2)}``.

    Empty / invalid pairs are ignored silently so the CLI is forgiving.
    """
    out: dict[str, float] = {}
    if not s:
        return out
    for kv in str(s).split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def _sample_weights(
    tr: pd.DataFrame,
    loss_mode: str,
    hard_paths: set[str] | None,
    hard_keys: set[str] | None = None,
    source_weights_overrides: dict[str, float] | None = None,
) -> np.ndarray | None:
    skip_weighted = loss_mode in ("balanced_sampler",)
    use_sampler = (
        not skip_weighted
        and (loss_mode in ("sampler_ce", "sampler_focal") or loss_mode.startswith("sampler_"))
    )
    if not use_sampler:
        return None
    counts = tr["label"].value_counts().to_dict()
    n0, n1 = int(counts.get(0, 1)), int(counts.get(1, 1))
    w_per_class = {0: 1.0 / max(1, n0), 1: 1.0 / max(1, n1)}
    sw = tr["label"].map(w_per_class).astype(np.float64)
    if "label_quality" in tr.columns:
        qmap = {"gold": 1.0, "silver": 0.85, "weak": 0.65}
        sw = sw * tr["label_quality"].map(qmap).fillna(0.75).astype(np.float64)
    # Hard-negative upweight (same semantics as before; complements extra draws in balanced_sampler)
    if hard_paths or hard_keys:
        hits = _hard_negative_hits(tr, hard_paths, hard_keys)
        sw = sw * (1.0 + 2.0 * hits.astype(np.float64))
        with contextlib.suppress(Exception):
            print(f"[train] hard_negative upweight matched rows: {int(hits.sum())}/{len(tr)}")
    # Source-aware sampling (multiplies class-balance weights)
    if "source" in tr.columns:
        src = tr["source"].astype(str)
        src_w_map: dict[str, float] = {
            "binary_root": 1.0,
            "flame3": 1.0,
            "flame_video_nofire": float(TRAIN_DEFAULT.get("flame_video_nofire_weight", 1.85)),
            # CART is auxiliary ground-domain negatives; downweight by default
            # so it does not dominate the no_fire pool of every batch.
            "cart_aux": 0.5,
        }
        if source_weights_overrides:
            for k, v in source_weights_overrides.items():
                src_w_map[str(k)] = float(v)
            print(f"[train] source_weights overrides applied: {source_weights_overrides}")
        sw = sw * src.map(lambda s: float(src_w_map.get(s, 1.0))).astype(np.float64)

    # Optional per-row weight (e.g. auxiliary downweight) if present
    if "sampling_weight" in tr.columns:
        w = pd.to_numeric(tr["sampling_weight"], errors="coerce").fillna(1.0).astype(np.float64)
        sw = sw * np.clip(w.values, 0.0, 10.0)
    return sw.values.astype(np.float32)


def _is_fusion_dual_family(mf: str) -> bool:
    return str(mf or "").lower() in FUSION_DUAL_FAMILIES


def _set_rgb_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith(("rgb_branch.", "rgb_fx.")):
            p.requires_grad = bool(trainable)


def build_optimizer_with_thermal_multiplier(
    model: torch.nn.Module,
    *,
    mf: str,
    lr: float,
    weight_decay: float,
    thermal_lr_mult: float,
) -> torch.optim.Optimizer:
    """AdamW with optional LR boost on thermal encoders."""
    mf_l = str(mf or "").lower()
    if not _is_fusion_dual_family(mf_l):
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rgb_p: list[torch.nn.Parameter] = []
    th_p: list[torch.nn.Parameter] = []
    rest_p: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("rgb_branch.") or name.startswith("rgb_fx."):
            rgb_p.append(p)
        elif name.startswith("th_branch.") or name.startswith("th_fx."):
            th_p.append(p)
        else:
            rest_p.append(p)
    groups: list[dict] = []
    lr_b = float(lr)
    wdec = float(weight_decay)
    tmul = float(thermal_lr_mult)
    if rgb_p:
        groups.append({"params": rgb_p, "lr": lr_b})
    if th_p:
        groups.append({"params": th_p, "lr": lr_b * tmul})
    if rest_p:
        groups.append({"params": rest_p, "lr": lr_b})
    if not groups:
        return torch.optim.AdamW(model.parameters(), lr=lr_b, weight_decay=wdec)
    return torch.optim.AdamW(groups, weight_decay=wdec)


def append_experiment_csv_row(csv_path: str | Path, row: dict) -> None:
    import csv as _csv

    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted(row.keys())
    new_file = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _gated_fusion_reg_loss(
    aux: dict,
    *,
    entropy_weight: float,
    min_thermal_weight: float,
    min_thermal_floor: float,
    balance_weight: float,
) -> torch.Tensor:
    """Auxiliary losses for ``DualBranchGatedFusion`` to reduce RGB-only gate collapse."""
    g_r = aux["gate_rgb"]
    g_t = aux["gate_thermal"]
    device = g_r.device
    dtype = g_r.dtype
    total = torch.zeros((), device=device, dtype=dtype)
    if entropy_weight > 0:
        st = torch.stack([g_r.clamp_min(1e-8), g_t.clamp_min(1e-8)], dim=1)
        ent = -(st * st.log()).sum(dim=1).mean()
        total = total - float(entropy_weight) * ent
    if min_thermal_weight > 0 and float(min_thermal_floor) > 0:
        pen = torch.relu(float(min_thermal_floor) - g_t).pow(2).mean()
        total = total + float(min_thermal_weight) * pen
    if balance_weight > 0:
        total = total + float(balance_weight) * (g_r - g_t).pow(2).mean()
    return total.float()


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
    grad_accum_steps: int = 1,
    exclude_sources: list[str] | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    thermal_norm: str | None = None,
    flame_video_nofire_weight: float | None = None,
    inference_threshold: float | None = None,
    no_fire_weight: float = 1.0,
    fire_weight: float = 1.0,
    selection_metric: str = "realistic",
    source_weights: str | dict | None = None,
    modal_dropout_p: float = 0.0,
    thermal_init: str = "mean_rgb",
    freeze_rgb_epochs: int = 0,
    thermal_lr_mult: float = 1.0,
    label_smoothing: float = 0.05,
    balanced_thermal_aug: bool = True,
    experiment_log_csv: str | None = None,
    experiment_name: str | None = None,
    gate_entropy_weight: float = 0.0,
    gate_min_thermal_floor: float = 0.0,
    gate_min_thermal_weight: float = 0.0,
    gate_balance_weight: float = 0.0,
    rgb_aug_intensity: float = 1.15,
    thermal_aug_intensity: float = 1.0,
):
    sel_norm_o = str(selection_metric).strip().lower()
    if sel_norm_o in ("protocol_balanced", "stress", "clean"):
        print(f"[train] selection_metric={selection_metric!r} is deprecated → using realistic")
        selection_metric = "realistic"
    else:
        selection_metric = sel_norm_o
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()
    nw = float(no_fire_weight)
    fw = float(fire_weight)
    df = _load_index(csv_path)
    df = _prepare_df(df)
    if exclude_sources:
        ex = {str(s).strip() for s in exclude_sources if str(s).strip()}
        if ex:
            before = len(df)
            df = df[~df["source"].astype(str).isin(ex)].copy()
            print(f"[train] exclude_sources={sorted(ex)} removed={before-len(df)} kept={len(df)}")

    mf = (model_family or "early_fusion").lower()
    if mf == "rgb_baseline":
        mode = "rgb"
    elif mf == "thermal_baseline":
        mode = "thermal"
    elif mf == "early_fusion" or _is_fusion_dual_family(mf):
        mode = "fusion"

    df, drop_tr = filter_df_existing_paths(df, mode=mode)
    if drop_tr:
        print(f"[train] Dropped {drop_tr} rows (missing files on disk for mode={mode!r})")

    tr, va, test_df, extra_test_df = _split_data(df, extra_test_ratio, val_split, flame_test_ratio)
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    extra_test_df = extra_test_df.reset_index(drop=True)

    thermal_norm = str(thermal_norm) if thermal_norm is not None else str(TRAIN_DEFAULT.get("thermal_norm", "percentile"))
    thermal_mu_es = thermal_sig_es = None
    if thermal_norm.strip().lower() == "train_zscore":
        from ..data.thermal_stats import estimate_thermal_mu_sigma

        thermal_mu_es, thermal_sig_es = estimate_thermal_mu_sigma(
            tr, path_col="path_th", max_samples=min(800, len(tr))
        )
        print(
            f"[train] thermal train_zscore from TRAIN split: mu={thermal_mu_es:.6g} "
            f"sigma={thermal_sig_es:.6g}"
        )

    print(f"\n[{mode}/{mf}] train={len(tr)} | val={len(va)} | test={len(test_df)} | extra_test={len(extra_test_df)}")
    label_train = _label_counts_dict(tr)
    label_val = _label_counts_dict(va)
    label_test = _label_counts_dict(test_df)
    print(f"[train] label distribution train={label_train} val={label_val} test={label_test}")
    print(f"[train] source row counts train={_source_counts_breakdown(tr)}")
    print(f"[train] source row counts val={_source_counts_breakdown(va)}")
    print(f"[train] source row counts test={_source_counts_breakdown(test_df)}")
    print(f"[train] per-source labels train={_per_source_label_breakdown(tr)}")
    print(f"[train] per-source labels val={_per_source_label_breakdown(va)}")
    print(f"[train] per-source labels test={_per_source_label_breakdown(test_df)}")
    sel_norm = str(selection_metric or "realistic").lower()
    print(f"[train] selection_metric={sel_norm} no_fire_weight={nw} fire_weight={fw} loss_mode={loss_mode}")
    if float(modal_dropout_p) > 0.0:
        print(f"[train] modal_dropout_p={float(modal_dropout_p)} (fusion-only at input level)")

    # Sanity warnings (loud, non-fatal): mirror index-build warnings so users
    # who skipped the index step still see them.
    test_no_fire = int((test_df["label"] == 0).sum()) if "label" in test_df.columns else 0
    if test_no_fire < 200:
        print(
            f"[train][WARN] test no_fire={test_no_fire} (<200) — FPR / specificity numbers will be noisy."
        )
    for sp_name, sp_df in [("train", tr), ("val", va), ("test", test_df)]:
        if len(sp_df) and "source" in sp_df.columns:
            share = sp_df["source"].astype(str).value_counts(normalize=True)
            if len(share) and float(share.iloc[0]) > 0.70:
                print(
                    f"[train][WARN] source '{share.index[0]}' dominates {sp_name} "
                    f"({100.0 * float(share.iloc[0]):.1f}%) — global metrics reflect this source."
                )

    training_class_balance = {
        "train": label_train,
        "val": label_val,
        "test": label_test,
        "no_fire_weight": nw,
        "fire_weight": fw,
        "loss_mode": str(loss_mode),
        "balanced_batches": loss_mode == "balanced_sampler",
        "source_distribution": {
            "train": _source_counts_breakdown(tr),
            "val": _source_counts_breakdown(va),
            "test": _source_counts_breakdown(test_df),
        },
        "per_source_labels": {
            "train": _per_source_label_breakdown(tr),
            "val": _per_source_label_breakdown(va),
            "test": _per_source_label_breakdown(test_df),
        },
    }

    # Optional override for sampling weights experiments
    if flame_video_nofire_weight is not None:
        try:
            TRAIN_DEFAULT["flame_video_nofire_weight"] = float(flame_video_nofire_weight)
            print(f"[train] override flame_video_nofire_weight={float(flame_video_nofire_weight)}")
        except Exception:
            pass

    hard_paths: set[str] | None = None
    hard_keys: set[str] | None = None
    if hard_negative_csv and Path(hard_negative_csv).exists():
        hdf = pd.read_csv(hard_negative_csv)
        hp: set[str] = set()
        if "path_rgb" in hdf.columns:
            hp |= set(hdf["path_rgb"].astype(str).tolist())
        if "path_th" in hdf.columns:
            hp |= set(hdf["path_th"].astype(str).tolist())
        if not hp and len(hdf.columns) > 0:
            hp |= set(hdf[hdf.columns[0]].astype(str).tolist())
        hard_paths = hp or None
        if "key" in hdf.columns:
            hard_keys = set(hdf["key"].astype(str).tolist()) or None
        print(
            f"[train] hard_negative_csv: paths={len(hard_paths) if hard_paths else 0} "
            f"keys={len(hard_keys) if hard_keys else 0}"
        )

    in_ch, _ = get_model_config(mode)
    model = make_classifier(
        mf,
        backbone,
        mode,
        num_classes=2,
        pretrained=True,
        thermal_init=str(thermal_init or "mean_rgb"),
    ).to(device)
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_params = sum(int(p.numel()) for p in model.parameters())
    print(f"[train] params: trainable={trainable_params:,} total={total_params:,}")

    src_w_overrides = (
        source_weights
        if isinstance(source_weights, dict)
        else _parse_source_weights(source_weights)
    )
    sw_arr = _sample_weights(
        tr,
        loss_mode,
        hard_paths,
        hard_keys=hard_keys,
        source_weights_overrides=src_w_overrides,
    )
    sampler = (
        WeightedRandomSampler(
            weights=torch.tensor(sw_arr, dtype=torch.float32),
            num_samples=len(tr),
            replacement=True,
        )
        if sw_arr is not None
        else None
    )

    cpu_count = os.cpu_count() or 2
    default_workers = max(0, min(8, cpu_count - 1))
    num_workers = int(num_workers) if num_workers is not None else int(TRAIN_DEFAULT.get("num_workers", default_workers))
    persistent_workers = (
        bool(persistent_workers)
        if persistent_workers is not None
        else bool(TRAIN_DEFAULT.get("persistent_workers", num_workers > 0))
    )
    prefetch_factor = int(prefetch_factor) if prefetch_factor is not None else int(TRAIN_DEFAULT.get("prefetch_factor", 2))
    pin_memory = bool(pin_memory) if pin_memory is not None else bool(TRAIN_DEFAULT.get("pin_memory", use_cuda))
    loader_common = {
        "batch_size": bs,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory and use_cuda),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = max(1, prefetch_factor)

    print(
        f"[train] DataLoader config: workers={num_workers}, "
        f"pin_memory={loader_common['pin_memory']}, "
        f"prefetch={loader_common.get('prefetch_factor', 'n/a')}, "
        f"persistent={loader_common['persistent_workers']}, "
        f"thermal_norm={thermal_norm} rgb_aug={float(rgb_aug_intensity):.2f} th_aug={float(thermal_aug_intensity):.2f}"
    )
    eff_bs = int(bs) * max(1, int(grad_accum_steps))
    print(f"[train] effective_batch_size={eff_bs} (bs={int(bs)} x grad_accum_steps={int(max(1, grad_accum_steps))})")

    train_ds = FlameDataset(
        tr,
        mode=mode,
        size=size,
        train=True,
        thermal_norm=thermal_norm,
        thermal_mu=thermal_mu_es,
        thermal_sigma=thermal_sig_es,
        balanced_thermal_aug=balanced_thermal_aug,
        rgb_aug_intensity=float(rgb_aug_intensity),
        thermal_aug_intensity=float(thermal_aug_intensity),
    )
    if loss_mode == "balanced_sampler":
        extra0 = _extra_hard_negative_class0_indices(tr, hard_paths, hard_keys)
        if extra0:
            print(f"[train] balanced_sampler: extra class-0 index slots from hard negatives: {len(extra0)}")
        bsp = BalancedTwoClassBatchSampler(
            tr["label"].astype(int).to_numpy(),
            int(bs),
            extra_class0_indices=extra0 or None,
        )
        train_kw = {k: v for k, v in loader_common.items() if k != "batch_size"}
        train_loader = DataLoader(train_ds, batch_sampler=bsp, **train_kw)
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=(sampler is None),
            sampler=sampler,
            **loader_common,
        )
    val_loader = DataLoader(
        FlameDataset(
            va,
            mode=mode,
            size=size,
            train=False,
            thermal_norm=thermal_norm,
            thermal_mu=thermal_mu_es,
            thermal_sigma=thermal_sig_es,
            balanced_thermal_aug=False,
        ),
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        FlameDataset(
            test_df,
            mode=mode,
            size=size,
            train=False,
            thermal_norm=thermal_norm,
            thermal_mu=thermal_mu_es,
            thermal_sigma=thermal_sig_es,
            balanced_thermal_aug=False,
        ),
        shuffle=False,
        **loader_common,
    )
    extra_test_loader = None
    if len(extra_test_df) > 0:
        extra_test_loader = DataLoader(
            FlameDataset(
                extra_test_df,
                mode=mode,
                size=size,
                train=False,
                thermal_norm=thermal_norm,
                thermal_mu=thermal_mu_es,
                thermal_sigma=thermal_sig_es,
                balanced_thermal_aug=False,
            ),
            shuffle=False,
            **loader_common,
        )

    # Dataloader batch counts
    with contextlib.suppress(Exception):
        print(
            f"[train] loader batches: train={len(train_loader)} val={len(val_loader)} "
            f"test={len(test_loader)}"
            + (f" extra_test={len(extra_test_loader)}" if extra_test_loader is not None else "")
        )

    counts = tr["label"].value_counts().to_dict()
    n0, n1 = int(counts.get(0, 1)), int(counts.get(1, 1))
    w0 = (n0 + n1) / (2 * max(1, n0)) * nw
    w1 = (n0 + n1) / (2 * max(1, n1)) * fw
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    class_counts = torch.tensor([float(n0), float(n1)], dtype=torch.float32).to(device)
    ln = loss_name or (
        "ce"
        if loss_mode == "sampler_ce"
        else ("cb_focal" if loss_mode in ("class_balanced_focal", "balanced_sampler") else "focal")
    )
    loss_fn = build_loss(
        ln,
        class_weights,
        device,
        focal_gamma=focal_gamma,
        label_smoothing=float(label_smoothing or 0.05),
        class_counts=class_counts,
        manual_class_gain=(nw, fw),
    )
    loss_fn = loss_fn.to(device)

    # First batch shapes + memory snapshots (helps distinguish RAM vs CUDA OOM early)
    print("[smoke] memory snapshot BEFORE first batch:", {
        "ram": _ram_snapshot(),
        "gpu": _gpu_snapshot(),
    })
    _print_first_batch_info(train_loader, device=device, pin_memory=bool(loader_common.get("pin_memory", False)))
    print("[smoke] memory snapshot AFTER first batch:", {
        "ram": _ram_snapshot(),
        "gpu": _gpu_snapshot(),
    })

    wd = float(TRAIN_DEFAULT.get("weight_decay", 0.01))
    frz_rgb = int(max(0, int(freeze_rgb_epochs)))
    if frz_rgb > 0 and _is_fusion_dual_family(mf):
        _set_rgb_encoder_trainable(model, False)
        print(f"[train] freeze_rgb_epochs={frz_rgb} (RGB encoder frozen initially)")
    opt = build_optimizer_with_thermal_multiplier(
        model, mf=mf, lr=float(lr), weight_decay=wd, thermal_lr_mult=float(thermal_lr_mult)
    )
    if scheduler_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    # New torch.amp API (PyTorch 2.x); falls back to legacy torch.cuda.amp on older builds.
    if use_cuda and use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    use_gated_reg = (
        mf == "dual_branch_gated_fusion"
        and (
            float(gate_entropy_weight) > 0.0
            or float(gate_min_thermal_weight) > 0.0
            or float(gate_balance_weight) > 0.0
        )
    )
    if use_gated_reg:
        print(
            "[train] gated fusion regularizers: "
            f"w_ent={float(gate_entropy_weight):g} floor={float(gate_min_thermal_floor):g} "
            f"w_min_th={float(gate_min_thermal_weight):g} w_bal={float(gate_balance_weight):g}"
        )
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    # Tracks the best validation selection score (legacy or realistic, see sel_norm).
    best_val_score = -1.0
    best_recall_fpr_key: tuple | None = None
    patience_counter = 0

    grad_accum_steps = max(1, int(grad_accum_steps))
    log_path = Path(OUTPUTS_DIR) / f"train_log_{mode}_{mf}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        if frz_rgb > 0 and ep == frz_rgb + 1 and _is_fusion_dual_family(mf):
            print("[train] unfreezing RGB encoder — rebuilding optimizer + scheduler")
            _set_rgb_encoder_trainable(model, True)
            opt = build_optimizer_with_thermal_multiplier(
                model,
                mf=mf,
                lr=float(lr),
                weight_decay=wd,
                thermal_lr_mult=float(thermal_lr_mult),
            )
            if scheduler_kind == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(int(epochs) - int(ep) + 1, 1)
                )
            else:
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="max", factor=0.5, patience=2
                )

        model.train()
        pbar = tqdm(train_loader, desc=f"{mode} ep{ep}/{epochs}")
        opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(pbar, start=1):
            nb = bool(loader_common.get("pin_memory", False)) and device == "cuda"
            try:
                x = x.to(device, non_blocking=nb)
                y = y.to(device, non_blocking=nb)
            except RuntimeError as e:
                msg = str(e).lower()
                if "cuda out of memory" in msg or "out of memory" in msg:
                    print(f"[train][OOM][CUDA] at batch={bi}: {e}")
                    print("[train][OOM][CUDA] snapshot:", _gpu_snapshot())
                    raise
                raise
            except MemoryError as e:
                print(f"[train][OOM][RAM] at batch={bi}: {e}")
                print("[train][OOM][RAM] snapshot:", _ram_snapshot())
                raise

            # Modal dropout (fusion only): with prob ``modal_dropout_p`` zero out
            # one of the two modalities at the input level. Forces the fusion
            # head to not over-trust a single sensor (helps on sun reflections,
            # hot non-fire surfaces). Channels: 0:3 = RGB, 3: = thermal.
            if (
                mode == "fusion"
                and float(modal_dropout_p) > 0.0
                and x.dim() == 4
                and x.shape[1] >= 4
            ):
                if torch.rand((), device=x.device).item() < float(modal_dropout_p):
                    if torch.rand((), device=x.device).item() < 0.5:
                        x[:, :3] = 0.0
                    else:
                        x[:, 3:] = 0.0

            if scaler is not None:
                try:
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                except (TypeError, AttributeError):
                    autocast_ctx = torch.cuda.amp.autocast()
                with autocast_ctx:
                    if use_gated_reg:
                        logits, aux = model(x, return_aux=True)
                        loss_cls = loss_fn(logits, y) / grad_accum_steps
                        loss_reg = (
                            _gated_fusion_reg_loss(
                                aux,
                                entropy_weight=gate_entropy_weight,
                                min_thermal_weight=gate_min_thermal_weight,
                                min_thermal_floor=gate_min_thermal_floor,
                                balance_weight=gate_balance_weight,
                            )
                            / grad_accum_steps
                        )
                        loss = loss_cls + loss_reg
                    else:
                        logits = model(x)
                        loss = loss_fn(logits, y) / grad_accum_steps
                scaler.scale(loss).backward()
                if (bi % grad_accum_steps == 0) or (bi == len(train_loader)):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                if use_gated_reg:
                    logits, aux = model(x, return_aux=True)
                    loss_cls = loss_fn(logits, y) / grad_accum_steps
                    loss_reg = (
                        _gated_fusion_reg_loss(
                            aux,
                            entropy_weight=gate_entropy_weight,
                            min_thermal_weight=gate_min_thermal_weight,
                            min_thermal_floor=gate_min_thermal_floor,
                            balance_weight=gate_balance_weight,
                        )
                        / grad_accum_steps
                    )
                    loss = loss_cls + loss_reg
                else:
                    logits = model(x)
                    loss = loss_fn(logits, y) / grad_accum_steps
                loss.backward()
                if (bi % grad_accum_steps == 0) or (bi == len(train_loader)):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=float(loss.item() * grad_accum_steps))
            if max_train_batches is not None and bi >= int(max_train_batches):
                break

        # Allow fast smoke runs without any validation
        if max_val_batches == 0:
            print("[val] skipped (max_val_batches=0)")
            if scheduler_kind != "plateau":
                sched.step()
            # Without validation we cannot early-stop or checkpoint based on val
            continue

        va_eval = va
        corr_name, corr_sev = protocol_corruption(mode)
        ep_seed = int(ep) * 9973 + 42
        model.eval()

        if max_val_batches is not None:
            vy, val_logits = eval_logits_corrupted(
                model,
                val_loader,
                device,
                corruption_name=corr_name,
                severity=corr_sev,
                seed=ep_seed,
                max_batches=int(max_val_batches),
            )
            try:
                va_eval = va.iloc[: len(vy)].reset_index(drop=True)
            except Exception:
                va_eval = va
        else:
            vy, val_logits = eval_logits_corrupted(
                model,
                val_loader,
                device,
                corruption_name=corr_name,
                severity=corr_sev,
                seed=ep_seed,
                max_batches=None,
            )

        if len(vy) == 0:
            print("[val] protocol eval produced zero rows — skipping metric block for this epoch")
            if scheduler_kind != "plateau":
                sched.step()
            continue

        T = fit_temperature(vy, val_logits)
        vp = _probs_from_logits(val_logits, temperature=T)
        thr_f1 = find_best_threshold_f1(vy, vp)

        cand_thrs = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        sweep = [(t, metrics_at_threshold(vy, vp, float(t))) for t in cand_thrs]
        print(f"[val_realistic:{corr_name}@{corr_sev}] threshold sweep:")
        for t, m in sweep:
            print(
                f"  thr={t:.2f} spec={m.get('specificity', float('nan')):.3f} "
                f"fpr={m.get('false_positive_rate', float('nan')):.3f} "
                f"recall={m.get('recall', float('nan')):.3f} f1={m.get('f1', float('nan')):.3f}"
            )

        # Choose: among thresholds that keep recall near-best, minimize FPR, then maximize F1.
        max_rec = max(float(m.get("recall", 0.0)) for _, m in sweep)
        min_keep = max(0.0, max_rec - 0.02)  # allow up to 2% recall drop
        eligible = [(t, m) for (t, m) in sweep if float(m.get("recall", 0.0)) >= min_keep]
        if not eligible:
            eligible = sweep
        thr_oper = sorted(
            eligible,
            key=lambda tm: (
                float(tm[1].get("false_positive_rate", 1.0)),
                -float(tm[1].get("f1", 0.0)),
            ),
        )[0][0]
        thr_oper = float(thr_oper)
        vm_oper = metrics_at_threshold(vy, vp, thr_oper)
        print(
            f"[val_realistic] recommended_thr={thr_oper:.2f} (keep_recall>={min_keep:.3f}, "
            f"fpr={vm_oper.get('false_positive_rate', float('nan')):.3f}, "
            f"recall={vm_oper.get('recall', float('nan')):.3f})"
        )

        # Keep both: F1-optimal threshold for analysis, operating threshold for inference default.
        thr_default = float(thr_oper)
        if inference_threshold is not None:
            thr_default = float(inference_threshold)
            print(f"[val_realistic] inference_threshold override -> thr_default={thr_default:.2f}")
        best_thr = float(thr_default)
        vm = metrics_at_threshold(vy, vp, best_thr)
        ece = expected_calibration_error(vy, vp)
        brier = brier_score_binary(vy, vp)
        print(
            f"[val_realistic:{corr_name}@{corr_sev}] "
            f"acc={vm['acc']:.3f} bal_acc={vm['bal_acc']:.3f} auc={vm['auc']:.3f} ap={vm['ap']:.3f} "
            f"thr={best_thr:.2f} (thr_f1={thr_f1:.2f}) T={T:.3f} "
            f"P={vm['precision']:.3f} R={vm['recall']:.3f} F1={vm['f1']:.3f} "
            f"spec={vm.get('specificity', float('nan')):.3f} fpr={vm.get('false_positive_rate', float('nan')):.3f} "
            f"ECE={ece:.3f} Brier={brier:.3f}"
        )
        print("Val CM [[TN FP],[FN TP]]:\n", vm["cm"])

        if mf == "dual_branch_gated_fusion" and mode == "fusion":
            try:
                model.eval()
                xg, _ = next(iter(val_loader))
                nb = bool(loader_common.get("pin_memory", False)) and device == "cuda"
                xg = xg.to(device, non_blocking=nb)[: min(24, len(xg))]
                xg = CORRUPTIONS[str(corr_name)](xg, int(corr_sev))
                with torch.no_grad():
                    logits_g, aux_g = model(xg, return_aux=True)
                del logits_g
                if isinstance(aux_g, dict) and "gate_rgb" in aux_g:
                    gr = float(aux_g["gate_rgb"].mean().detach().cpu())
                    gt = float(aux_g["gate_thermal"].mean().detach().cpu())
                    print(f"[val] gated fusion mean gates (noisy mini-batch): RGB={gr:.4f} TH={gt:.4f}")
            except Exception as eg:
                print(f"[val] gated diagnostics skipped: {type(eg).__name__}: {eg}")

        ty, test_logits = eval_logits_corrupted(
            model,
            test_loader,
            device,
            corruption_name=corr_name,
            severity=corr_sev,
            seed=ep_seed + 3,
            max_batches=None,
        )
        tp = _probs_from_logits(test_logits, temperature=T)
        tm = metrics_at_threshold(ty, tp, best_thr)
        print(
            f"[test_realistic:{corr_name}@{corr_sev}] "
            f"acc={tm['acc']:.3f} bal_acc={tm['bal_acc']:.3f} auc={tm['auc']:.3f} ap={tm['ap']:.3f} "
            f"P={tm.get('precision', float('nan')):.3f} R={tm.get('recall', float('nan')):.3f} "
            f"F1={tm.get('f1', float('nan')):.3f} "
            f"spec={tm.get('specificity', float('nan')):.3f} fpr={tm.get('false_positive_rate', float('nan')):.3f}"
        )
        try:
            print("Test CM [[TN FP],[FN TP]]:\n", tm["cm"])
        except Exception:
            pass

        if extra_test_loader is not None:
            ey, extra_logits = eval_logits_corrupted(
                model,
                extra_test_loader,
                device,
                corruption_name=corr_name,
                severity=corr_sev,
                seed=ep_seed + 101,
                max_batches=None,
            )
            ep_probs = _probs_from_logits(extra_logits, temperature=T)
            pred = (ep_probs >= best_thr).astype(np.int64)
            n_fp = int((pred == 1).sum())
            n_tn = int((pred == 0).sum())
            fp_rate = n_fp / max(1, len(ey))
            print(
                f"Extra test noisy ({corr_name}@{corr_sev}) n={len(ey)} "
                f"FP={n_fp} TN={n_tn} FP_rate={fp_rate:.3f}"
            )

        if scheduler_kind == "plateau":
            sched.step(vm["ap"] if vm["ap"] == vm["ap"] else 0.0)
        else:
            sched.step()

        score_legacy = 0.5 * float(vm.get("f1", 0.0)) + 0.5 * float(vm.get("bal_acc", 0.0))
        score_realistic = realistic_selection_score(vm)
        key_recall_fpr = (
            recall_fpr_selection_key(vm, ece=float(ece), brier=float(brier))
            if sel_norm == "recall_fpr"
            else None
        )

        if sel_norm == "realistic":
            selection_score = score_realistic
        elif sel_norm == "recall_fpr" and key_recall_fpr is not None:
            selection_score = float(key_recall_fpr[1])
        else:
            selection_score = score_legacy

        if sel_norm == "recall_fpr":
            improved = key_recall_fpr is not None and (
                best_recall_fpr_key is None or key_recall_fpr > best_recall_fpr_key
            )
        else:
            improved = selection_score == selection_score and selection_score > best_val_score

        if improved:
            if sel_norm == "recall_fpr" and key_recall_fpr is not None:
                best_recall_fpr_key = key_recall_fpr
                best_val_score = float(key_recall_fpr[1])
            else:
                best_val_score = float(selection_score)
            patience_counter = 0
            thr_alarm_raw = _best_threshold_mode(vy, vp, "alarm")
            thr_review = _best_threshold_mode(vy, vp, "review")
            thr_alarm = max(float(THRESHOLD_ALARM_MIN), float(thr_alarm_raw))
            if thr_alarm != thr_alarm_raw:
                print(
                    f"[val_realistic] threshold_alarm clamped {thr_alarm_raw:.3f} -> {thr_alarm:.3f} "
                    f"(min={THRESHOLD_ALARM_MIN})"
                )

            def _metric_pack(m: dict) -> dict:
                return sanitize_for_json({k: v for k, v in m.items()})

            val_real_pack = _metric_pack(vm)
            tr_real_pack = _metric_pack(tm)
            ta_save = {
                "epochs": int(epochs),
                "batch_size": int(bs),
                "lr": float(lr),
                "loss_mode": str(loss_mode),
                "loss_name": str(ln),
                "scheduler_kind": str(scheduler_kind),
                "grad_accum_steps": int(grad_accum_steps),
                "no_fire_weight": float(nw),
                "fire_weight": float(fw),
                "training_class_balance": training_class_balance,
                "selection_metric": str(sel_norm),
                "thermal_init": str(thermal_init),
                "freeze_rgb_epochs": int(frz_rgb),
                "thermal_lr_mult": float(thermal_lr_mult),
                "modal_dropout_p": float(modal_dropout_p),
                "thermal_norm": str(thermal_norm),
                "thermal_mu": float(thermal_mu_es) if thermal_mu_es is not None else None,
                "thermal_sigma": float(thermal_sig_es) if thermal_sig_es is not None else None,
                "rgb_aug_intensity": float(rgb_aug_intensity),
                "thermal_aug_intensity": float(thermal_aug_intensity),
                "gate_entropy_weight": float(gate_entropy_weight),
                "gate_min_thermal_floor": float(gate_min_thermal_floor),
                "gate_min_thermal_weight": float(gate_min_thermal_weight),
                "gate_balance_weight": float(gate_balance_weight),
                "backbone": str(backbone),
            }

            torch.save(
                {
                    "mode": mode,
                    "model_family": mf,
                    "in_ch": in_ch,
                    "backbone": backbone,
                    "input_size": int(size),
                    "class_mapping": {"0": "no_fire", "1": "fire"},
                    "training_args": ta_save,
                    "state": model.state_dict(),
                    "threshold": float(best_thr),
                    "threshold_f1": float(thr_f1),
                    "threshold_recommended": float(thr_oper),
                    "threshold_alarm": float(thr_alarm),
                    "threshold_alarm_raw": float(thr_alarm_raw),
                    "threshold_alarm_clamped": float(thr_alarm),
                    "threshold_alarm_min": float(THRESHOLD_ALARM_MIN),
                    "threshold_review": float(thr_review),
                    "val_ap": float(vm["ap"]),
                    "val_score_f1_balacc": float(score_legacy),
                    "val_score_realistic": float(score_realistic),
                    "val_selection_score": float(selection_score),
                    "val_best_recall_fpr_key": (
                        list(best_recall_fpr_key) if best_recall_fpr_key is not None else None
                    ),
                    "temperature": float(T),
                    "eval_protocol_corruption": f"{corr_name}@{corr_sev}",
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                out_ckpt,
            )
            print(
                f"Saved (best sel={sel_norm} selection_score={selection_score:.4f}, "
                f"f1+balacc_legacy={score_legacy:.4f}, realistic={score_realistic:.4f}, T={T:.3f}): {out_ckpt}"
            )
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
                "training_args": ta_save,
                "threshold": float(best_thr),
                "temperature": float(T),
                "eval_protocol_corruption": f"{corr_name}@{corr_sev}",
                "val_score_f1_balacc": float(score_legacy),
                "val_score_realistic": float(score_realistic),
                "val_selection_score": float(selection_score),
                "selection_metric": str(sel_norm),
                "val_realistic": val_real_pack,
                "test_realistic": tr_real_pack,
                "val": val_real_pack,
                "test": tr_real_pack,
                "test_noisy": tr_real_pack,
                **extra_info,
            }
            metrics_path = Path(OUTPUTS_DIR) / f"metrics_{mode}_{mf}.json"
            if mf == "early_fusion" and mode == "fusion":
                metrics_path = Path(OUTPUTS_DIR) / f"metrics_{mode}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(metrics), f, indent=2, ensure_ascii=False)

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
                print(
                    f"Early stop ({sel_norm} selection score did not improve for {patience} epochs)"
                )
                break

        epoch_row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "epoch": int(ep),
            "mode": mode,
            "model_family": mf,
            "val_ap": float(vm.get("ap", float("nan"))),
            "val_f1": float(vm.get("f1", float("nan"))),
            "val_bal_acc": float(vm.get("bal_acc", float("nan"))),
            "val_score_f1_balacc": float(score_legacy),
            "val_score_realistic": float(score_realistic),
            "val_selection_score": float(selection_score),
            "selection_metric": str(sel_norm),
            "val_auc": float(vm.get("auc", float("nan"))),
            "best_val_score": float(best_val_score),
            "temperature": float(T),
            "threshold": float(best_thr),
            "lr": float(opt.param_groups[0]["lr"]),
            "grad_accum_steps": int(grad_accum_steps),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_row, ensure_ascii=False) + "\n")

    if calibrate_report and Path(out_ckpt).exists():
        try:
            ck = torch.load(out_ckpt, map_location=device, weights_only=True)
        except TypeError:
            ck = torch.load(out_ckpt, map_location=device)
        T = float(ck.get("temperature", 1.0))
        print(
            f"[calibrate] checkpoint temperature={T:.4f} "
            f"thresholds alarm={ck.get('threshold_alarm')} "
            f"alarm_raw={ck.get('threshold_alarm_raw')} "
            f"alarm_clamped={ck.get('threshold_alarm_clamped')} "
            f"review={ck.get('threshold_review')}"
        )

    # Final report: fixed thresholds under eval-time protocol corruption only.
    try:
        if Path(out_ckpt).exists():
            try:
                ck = torch.load(out_ckpt, map_location=device, weights_only=True)
            except TypeError:
                ck = torch.load(out_ckpt, map_location=device)
            model.load_state_dict(ck["state"])
            model.eval()
            T_best = float(ck.get("temperature", 1.0))
            cn, cs = protocol_corruption(mode)
            fixed = [0.50, 0.55]
            fin_seed = 900_001
            vy2, lg2 = eval_logits_corrupted(
                model,
                val_loader,
                device,
                corruption_name=cn,
                severity=cs,
                seed=fin_seed,
                max_batches=None,
            )
            ty2, lt2 = eval_logits_corrupted(
                model,
                test_loader,
                device,
                corruption_name=cn,
                severity=cs,
                seed=fin_seed + 7,
                max_batches=None,
            )
            vp2 = _probs_from_logits(lg2, temperature=T_best)
            tp2 = _probs_from_logits(lt2, temperature=T_best)
            print(f"[final] threshold comparison (val/test, {cn}@{cs}):")
            for t in fixed:
                mv = metrics_at_threshold(vy2, vp2, float(t))
                mt = metrics_at_threshold(ty2, tp2, float(t))
                print(
                    f"  thr={t:.2f} | "
                    f"VAL spec={mv.get('specificity', float('nan')):.3f} fpr={mv.get('false_positive_rate', float('nan')):.3f} "
                    f"recall={mv.get('recall', float('nan')):.3f} f1={mv.get('f1', float('nan')):.3f} || "
                    f"TEST spec={mt.get('specificity', float('nan')):.3f} fpr={mt.get('false_positive_rate', float('nan')):.3f} "
                    f"recall={mt.get('recall', float('nan')):.3f} f1={mt.get('f1', float('nan')):.3f}"
                )
    except Exception as e:
        print(f"[final] threshold comparison skipped: {type(e).__name__}: {e}")

    if experiment_log_csv:
        mp = Path(OUTPUTS_DIR) / f"metrics_{mode}_{mf}.json"
        row: dict = {
            "ts_done": datetime.now(timezone.utc).isoformat(),
            "experiment_name": (experiment_name or "").strip(),
            "csv_index": str(csv_path),
            "model_family": mf,
            "mode": mode,
            "backbone": str(backbone),
            "thermal_norm": str(thermal_norm),
            "thermal_init": str(thermal_init),
            "freeze_rgb_epochs": int(frz_rgb),
            "thermal_lr_mult": float(thermal_lr_mult),
            "modal_dropout_p": float(modal_dropout_p),
            "rgb_aug_intensity": float(rgb_aug_intensity),
            "thermal_aug_intensity": float(thermal_aug_intensity),
            "gate_entropy_weight": float(gate_entropy_weight),
            "gate_min_thermal_floor": float(gate_min_thermal_floor),
            "gate_min_thermal_weight": float(gate_min_thermal_weight),
            "gate_balance_weight": float(gate_balance_weight),
            "selection_metric": str(sel_norm),
            "loss_name": str(ln),
            "out_ckpt": str(out_ckpt),
        }
        try:
            if mp.exists():
                lastm = json.loads(mp.read_text(encoding="utf-8"))

                def _prot_fpr_triple(src: dict | None, row_d: dict, prefix: str) -> None:
                    if not isinstance(src, dict):
                        return
                    for short, sk in (
                        ("f1", "f1"),
                        ("recall", "recall"),
                        ("fpr", "false_positive_rate"),
                    ):
                        if sk in src:
                            v0 = src[sk]
                            row_d[f"{prefix}_{short}"] = (
                                float(v0) if not isinstance(v0, (list, dict)) else json.dumps(v0)
                            )

                vr = lastm.get("val_realistic")
                if not isinstance(vr, dict):
                    vr = lastm.get("val")
                _prot_fpr_triple(vr if isinstance(vr, dict) else None, row, "val_realistic")
                trb = (
                    lastm.get("test_realistic")
                    if isinstance(lastm.get("test_realistic"), dict)
                    else lastm.get("test_noisy")
                )
                if not isinstance(trb, dict):
                    trb = lastm.get("test")
                _prot_fpr_triple(trb if isinstance(trb, dict) else None, row, "test_realistic")
            append_experiment_csv_row(str(experiment_log_csv), row)
            print(f"[train] experiment log -> {experiment_log_csv}")
        except Exception as ex:
            print(f"[train] experiment_log_csv failed: {type(ex).__name__}: {ex}")

    return out_ckpt
