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
    metrics_per_source,
    sanitize_for_json,
    select_threshold_policies,
    realistic_selection_score,
    threshold_sweep_grid,
)
from .losses import build_loss
from .metrics import (
    eval_probs,
    eval_logits,
    metrics_at_threshold,
    find_best_threshold_f1,
    fit_temperature,
    expected_calibration_error,
    brier_score_binary,
    reliability_report,
    _best_threshold_mode,
)
from ..data import FlameDataset
from .robustness_eval import (
    augment_metrics_json_with_robustness_outputs,
    flame3_eval_slice,
    merge_robustness_into_metrics_json,
    run_flame3_robustness_evaluation,
)
from .sequence_metrics import compute_sequence_alarm_summary
from ..data.path_filter import filter_df_existing_paths
from ..data.split import _group_split_three_way, split_train_val_extra
from ..models import make_classifier, get_model_config

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


def _slug_path(x: object) -> str:
    return str(x).strip().replace("\\", "/").lower()


def _hard_negative_hit_series(
    tr: pd.DataFrame,
    path_slugs: set[str] | None,
    hard_keys: set[str] | None,
) -> pd.Series:
    """Boolean mask: row matches slug-normalized RGB/thermal paths and/or keys."""
    hits = pd.Series(False, index=tr.index)
    if path_slugs:
        if "path_rgb" in tr.columns:
            hits |= tr["path_rgb"].map(_slug_path).isin(path_slugs)
        if "path_th" in tr.columns:
            hits |= tr["path_th"].map(_slug_path).isin(path_slugs)
    if hard_keys and "key" in tr.columns:
        hits |= tr["key"].astype(str).isin(hard_keys)
    return hits


def _hard_negative_hits(tr: pd.DataFrame, hard_paths: set[str] | None, hard_keys: set[str] | None) -> pd.Series:
    slugs = {_slug_path(p) for p in (hard_paths or set())}
    return _hard_negative_hit_series(tr, slugs if slugs else None, hard_keys)


def _extra_hard_negative_class0_indices(
    tr: pd.DataFrame,
    path_slugs: set[str] | None,
    hard_keys: set[str] | None,
    *,
    hard_negative_weight: float = 2.0,
) -> list[int]:
    """Duplicate class-0 row indices so hard negatives appear proportionally more in balanced_sampler."""
    if not path_slugs and not hard_keys:
        return []
    hn = _hard_negative_hit_series(tr, path_slugs, hard_keys)
    lab = tr["label"].astype(int).to_numpy()
    m = hn.to_numpy(dtype=bool) & (lab == 0)
    base = np.flatnonzero(m).astype(np.int64).tolist()
    if not base:
        return []
    nw = max(1.0, float(hard_negative_weight))
    n_extra = max(0, int(round(nw)) - 1)
    out: list[int] = []
    for i in base:
        out.extend([int(i)] * n_extra)
    return out


def _apply_hard_negative_csv_enrichment(
    hard_negative_csv: str | None,
    tr: pd.DataFrame,
    extra_test_df: pd.DataFrame,
    *,
    mode: str,
    hard_negative_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str], dict[str, int]]:
    """Match/inject CSV hard negatives into ``tr`` and drop overlapping rows from ``extra_test``."""
    empty_stats = {"loaded": 0, "matched": 0, "injected": 0, "excluded_extra": 0}
    if not hard_negative_csv or not Path(hard_negative_csv).is_file():
        return tr, extra_test_df, set(), set(), empty_stats

    try:
        hdf = pd.read_csv(hard_negative_csv)
    except Exception:
        print(f"[hard-negative] unreadable CSV {hard_negative_csv!r}; skip", flush=True)
        return tr, extra_test_df, set(), set(), empty_stats

    n_csv = len(hdf)
    ex_slug: set[str] = set()
    ex_keys: set[str] = set()
    for _, r in hdf.iterrows():
        if "path_rgb" in hdf.columns:
            v = str(r.get("path_rgb", "")).strip()
            if v and v.lower() != "nan":
                ex_slug.add(_slug_path(v))
        if "path_th" in hdf.columns:
            v = str(r.get("path_th", "")).strip()
            if v and v.lower() != "nan":
                ex_slug.add(_slug_path(v))
        if "key" in hdf.columns:
            k = str(r.get("key", "")).strip()
            if k and k.lower() != "nan":
                ex_keys.add(k)

    excluded_extra = 0
    if len(extra_test_df) > 0 and (ex_slug or ex_keys):
        bad = pd.Series(False, index=extra_test_df.index)
        if "path_rgb" in extra_test_df.columns and ex_slug:
            bad |= extra_test_df["path_rgb"].map(_slug_path).isin(ex_slug)
        if "path_th" in extra_test_df.columns and ex_slug:
            bad |= extra_test_df["path_th"].map(_slug_path).isin(ex_slug)
        if "key" in extra_test_df.columns and ex_keys:
            bad |= extra_test_df["key"].astype(str).isin(ex_keys)
        excluded_extra = int(bad.sum())
        extra_out = extra_test_df.loc[~bad.to_numpy()].reset_index(drop=True)
        if excluded_extra > 0:
            print(
                f"[extra_test] excluded {excluded_extra} hard-negative training rows from evaluation",
                flush=True,
            )
    else:
        extra_out = extra_test_df

    tr_rows = tr.reset_index(drop=True).copy()
    tr_slug_rgb = set(tr_rows["path_rgb"].map(_slug_path)) if "path_rgb" in tr_rows.columns else set()
    tr_slug_th = set(tr_rows["path_th"].map(_slug_path)) if "path_th" in tr_rows.columns else set()
    tr_keys = set(tr_rows["key"].astype(str).tolist()) if "key" in tr_rows.columns else set()

    matched_csv = 0
    injected_rows: list[dict] = []
    hn_w = float(hard_negative_weight)

    for _, r in hdf.iterrows():
        key_csv = ""
        if "key" in hdf.columns:
            kk = str(r.get("key", "")).strip()
            if kk and kk.lower() != "nan":
                key_csv = kk

        matched_row = bool(key_csv and key_csv in tr_keys)
        pr_raw = ""
        pt_raw = ""
        if "path_rgb" in hdf.columns:
            pr_raw = str(r.get("path_rgb", "")).strip()
            if pr_raw.lower() == "nan":
                pr_raw = ""
        if "path_th" in hdf.columns:
            pt_raw = str(r.get("path_th", "")).strip()
            if pt_raw.lower() == "nan":
                pt_raw = ""

        srg = _slug_path(pr_raw) if pr_raw else ""
        stg = _slug_path(pt_raw) if pt_raw else ""
        if not matched_row and srg and srg in tr_slug_rgb:
            matched_row = True
        if not matched_row and stg and stg in tr_slug_th:
            matched_row = True

        if matched_row:
            matched_csv += 1
            continue

        if mode != "fusion" or not pr_raw or not pt_raw:
            continue
        p_r = Path(pr_raw.replace("\\", "/"))
        p_t = Path(pt_raw.replace("\\", "/"))
        if not p_r.is_file() or not p_t.is_file():
            continue
        if srg in tr_slug_rgb or stg in tr_slug_th:
            continue

        nk = key_csv if key_csv else f"extra_hard_negative_{srg}_{stg}"[:260]
        if nk in tr_keys:
            nk = nk + "__inj"
        sg = ("extra_hard_negative_" + (key_csv[:48] if key_csv else srg[-32:])).strip("_")

        injected_rows.append(
            {
                "path_rgb": str(p_r),
                "path_th": str(p_t),
                "label": 0,
                "label_fire": 0,
                "source": "extra_hard_negative",
                "split": "train",
                "key": nk,
                "split_group": sg,
                "used_for_hard_negative_training": True,
                "sampling_weight": hn_w,
            }
        )
        tr_slug_rgb.add(srg)
        tr_slug_th.add(stg)
        tr_keys.add(nk)

    k_inj = len(injected_rows)
    if injected_rows:
        tr_rows = pd.concat([tr_rows, pd.DataFrame(injected_rows)], ignore_index=True)

    print(f"[hard-negative] loaded {n_csv} rows", flush=True)
    print(f"[hard-negative] matched {matched_csv} existing rows", flush=True)
    print(f"[hard-negative] injected {k_inj} rows", flush=True)
    print(f"[hard-negative] weight={float(hard_negative_weight)}", flush=True)

    stats = {"loaded": n_csv, "matched": matched_csv, "injected": k_inj, "excluded_extra": excluded_extra}
    return tr_rows, extra_out, set(ex_slug), set(ex_keys), stats


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

    When the index has column ``split``, rows with ``train`` / ``val`` / ``test`` / ``extra_test`` are routed
    accordingly; ``extra_test`` is held out (used for ``extra_test_df`` and optional robustness slices).
    Rows with any other normalized split value raise ``ValueError``.
    """
    rng = np.random.default_rng(random_state)
    df = _prepare_df(df)

    # If master index already provides an explicit split, respect it.
    if "split" in df.columns:
        sp = df["split"].astype(str).str.lower().str.strip()
        ok = sp.isin(["train", "val", "test", "extra_test"])
        if int(ok.sum()) > 0:
            df2 = df.copy()
            df2["split"] = sp.where(ok, "")
            tr = df2[df2["split"] == "train"].copy()
            va = df2[df2["split"] == "val"].copy()
            test_df = df2[df2["split"] == "test"].copy()
            extra_test_df = df2[df2["split"] == "extra_test"].copy()
            orphaned = df2[~df2["split"].isin(["train", "val", "test", "extra_test"])]
            if len(orphaned) > 0:
                raise ValueError(
                    f"[train] {len(orphaned)} rows have split not in train|val|test|extra_test (after normalizing authoritative splits). "
                    "Fix master_index or filtering."
                )
            return (
                tr.reset_index(drop=True),
                va.reset_index(drop=True),
                test_df.reset_index(drop=True),
                extra_test_df.reset_index(drop=True),
            )

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
    hard_path_slugs: set[str] | None,
    hard_keys: set[str] | None = None,
    source_weights_overrides: dict[str, float] | None = None,
    *,
    hard_negative_weight: float = 2.0,
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
    # Hard-negative upweight (sampler modes only; balanced_sampler uses duplicate indices)
    if hard_path_slugs or hard_keys:
        hits_arr = _hard_negative_hit_series(tr, hard_path_slugs, hard_keys)
        sw = sw * np.where(
            hits_arr.to_numpy(dtype=bool),
            np.float64(max(1.0, float(hard_negative_weight))),
            np.float64(1.0),
        )
        with contextlib.suppress(Exception):
            print(f"[train] hard_negative upweight matched rows: {int(hits_arr.sum())}/{len(tr)}")
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
            "extra_hard_negative": 1.0,
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
    selection_metric: str = "f1_balacc",
    source_weights: str | dict | None = None,
    modal_dropout_p: float = 0.0,
    robustness_eval: bool = False,
    hard_negative_weight: float = 2.0,
    aug_strength: str = "default",
):
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
    elif mf in ("early_fusion", "dual_branch_fusion"):
        mode = "fusion"

    df, drop_tr = filter_df_existing_paths(df, mode=mode)
    if drop_tr:
        print(f"[train] Dropped {drop_tr} rows (missing files on disk for mode={mode!r})")

    tr, va, test_df, extra_test_df = _split_data(df, extra_test_ratio, val_split, flame_test_ratio)
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    extra_test_df = extra_test_df.reset_index(drop=True)

    tr, extra_test_df, path_slug_hn, keys_hn, _hn_dbg = _apply_hard_negative_csv_enrichment(
        hard_negative_csv,
        tr,
        extra_test_df,
        mode=mode,
        hard_negative_weight=float(hard_negative_weight),
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
    sel_norm = str(selection_metric or "f1_balacc").lower()
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

    in_ch, _ = get_model_config(mode)
    model = make_classifier(mf, backbone, mode, num_classes=2, pretrained=True).to(device)
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
        path_slug_hn if path_slug_hn else None,
        hard_keys=keys_hn if keys_hn else None,
        source_weights_overrides=src_w_overrides,
        hard_negative_weight=float(hard_negative_weight),
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
    thermal_norm = str(thermal_norm) if thermal_norm is not None else str(TRAIN_DEFAULT.get("thermal_norm", "percentile"))
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
        f"thermal_norm={thermal_norm}"
    )
    eff_bs = int(bs) * max(1, int(grad_accum_steps))
    print(f"[train] effective_batch_size={eff_bs} (bs={int(bs)} x grad_accum_steps={int(max(1, grad_accum_steps))})")

    aug_str_norm = str(aug_strength or "default").strip().lower()
    print(f"[train] aug_strength={aug_str_norm}")
    train_ds = FlameDataset(
        tr,
        mode=mode,
        size=size,
        train=True,
        thermal_norm=thermal_norm,
        aug_strength=aug_str_norm,
    )
    if loss_mode == "balanced_sampler":
        extra0 = _extra_hard_negative_class0_indices(
            tr,
            path_slug_hn if path_slug_hn else None,
            keys_hn if keys_hn else None,
            hard_negative_weight=float(hard_negative_weight),
        )
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
        FlameDataset(va, mode=mode, size=size, train=False, thermal_norm=thermal_norm),
        shuffle=False,
        **loader_common,
    )
    test_loader = DataLoader(
        FlameDataset(test_df, mode=mode, size=size, train=False, thermal_norm=thermal_norm),
        shuffle=False,
        **loader_common,
    )
    extra_test_loader = None
    if len(extra_test_df) > 0:
        extra_test_loader = DataLoader(
            FlameDataset(extra_test_df, mode=mode, size=size, train=False, thermal_norm=thermal_norm),
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if scheduler_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if (use_cuda and use_amp) else None

    out_ckpt = out_ckpt or str(MODELS_DIR / f"{mode}.pt")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    best_val_ap = -1.0
    patience_counter = 0

    hn_excluded_xt = int(_hn_dbg.get("excluded_extra", 0))

    grad_accum_steps = max(1, int(grad_accum_steps))
    log_path = Path(OUTPUTS_DIR) / f"train_log_{mode}_{mf}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
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
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = loss_fn(logits, y) / grad_accum_steps
                scaler.scale(loss).backward()
                if (bi % grad_accum_steps == 0) or (bi == len(train_loader)):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
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
        if max_val_batches is not None:
            # Build a small val loader slice by iterating a few batches
            model.eval()
            ys, logits_list = [], []
            with torch.no_grad():
                for vi, (xv, yv) in enumerate(val_loader, start=1):
                    nb = bool(loader_common.get("pin_memory", False)) and device == "cuda"
                    xv = xv.to(device, non_blocking=nb)
                    logits = model(xv)
                    logits_list.append(logits.cpu().numpy())
                    ys.extend(yv.numpy().tolist())
                    if vi >= int(max_val_batches):
                        break
            vy_log = np.asarray(ys, dtype=np.int64)
            val_logits = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 2), dtype=np.float32)
            # Keep a matching slice of `va` so per-source metrics / FP export don't mismatch lengths.
            try:
                va_eval = va.iloc[: len(vy_log)].reset_index(drop=True)
            except Exception:
                va_eval = va
        else:
            vy_log, val_logits = eval_logits(model, val_loader, device)
        T = fit_temperature(vy_log, val_logits)
        vy = vy_log
        vp = _probs_from_logits(val_logits, temperature=T)
        thr_f1 = find_best_threshold_f1(vy, vp)

        # Threshold sweep for operating point (optimize low FPR without killing recall)
        # Include lower thresholds for recall-focused operating points
        cand_thrs = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        sweep = [(t, metrics_at_threshold(vy, vp, float(t))) for t in cand_thrs]
        print("[val] threshold sweep (operating point):")
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
            f"[val] recommended_thr={thr_oper:.2f} (keep_recall>={min_keep:.3f}, "
            f"fpr={vm_oper.get('false_positive_rate', float('nan')):.3f}, "
            f"recall={vm_oper.get('recall', float('nan')):.3f})"
        )

        # Keep both: F1-optimal threshold for analysis, operating threshold for inference default.
        thr_default = float(thr_oper)
        if inference_threshold is not None:
            thr_default = float(inference_threshold)
            print(f"[val] inference_threshold override -> thr_default={thr_default:.2f}")
        best_thr = float(thr_default)
        vm = metrics_at_threshold(vy, vp, best_thr)
        ece = expected_calibration_error(vy, vp)
        brier = brier_score_binary(vy, vp)
        print(
            f"Val acc={vm['acc']:.3f} bal_acc={vm['bal_acc']:.3f} auc={vm['auc']:.3f} ap={vm['ap']:.3f} "
            f"thr={best_thr:.2f} (thr_f1={thr_f1:.2f}) T={T:.3f} "
            f"P={vm['precision']:.3f} R={vm['recall']:.3f} F1={vm['f1']:.3f} "
            f"spec={vm.get('specificity', float('nan')):.3f} fpr={vm.get('false_positive_rate', float('nan')):.3f} "
            f"ECE={ece:.3f} Brier={brier:.3f}"
        )
        print("Val CM [[TN FP],[FN TP]]:\n", vm["cm"])

        # Per-source validation metrics
        per_source: dict[str, dict] = {}
        if "source" in va_eval.columns:
            srcs = va_eval["source"].astype(str).to_numpy()
            for s in sorted(set(srcs.tolist())):
                m = srcs == s
                if int(m.sum()) < 5:
                    continue
                # Skip per-source metrics if that slice has only one class (prevents sklearn warnings)
                if len(set(np.asarray(vy[m], dtype=np.int64).tolist())) < 2:
                    continue
                ms = metrics_at_threshold(vy[m], vp[m], best_thr)
                per_source[s] = {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in ms.items()}
            if per_source:
                print(
                    "[val] per_source metrics (thr shared):",
                    {
                        k: {
                            "f1": round(v["f1"], 3),
                            "bal_acc": round(v["bal_acc"], 3),
                            "n": int((va_eval["source"].astype(str) == k).sum()),
                        }
                        for k, v in per_source.items()
                    },
                )

        # Save false positives from validation (y=0, pred=1)
        pred_val = (vp >= float(best_thr)).astype(np.int64)
        fp_mask = (vy == 0) & (pred_val == 1)
        if int(fp_mask.sum()) > 0:
            fp_df = va_eval.loc[fp_mask].copy()
            fp_df["prob_fire"] = vp[fp_mask].astype(np.float32)
            # Expose label columns so downstream hard-negative retraining can
            # distinguish genuine negatives (label=0) from mis-labelled rows.
            if "label" not in fp_df.columns:
                fp_df["label"] = 0
            if "label_fire" not in fp_df.columns and "label" in fp_df.columns:
                fp_df["label_fire"] = fp_df["label"].astype(int)
            keep_cols = [
                c
                for c in [
                    "source",
                    "path_rgb",
                    "path_th",
                    "prob_fire",
                    "label",
                    "label_fire",
                    "key",
                    "split_group",
                ]
                if c in fp_df.columns
            ]
            fp_out = Path(OUTPUTS_DIR) / f"val_false_positives_{mode}_{mf}.csv"
            fp_df[keep_cols].to_csv(fp_out, index=False)
            print(f"[val] false_positives saved: {fp_out} (n={int(fp_mask.sum())})")

        ty, tp = eval_probs(model, test_loader, device, temperature=T)
        tm = metrics_at_threshold(ty, tp, best_thr)
        print(
            f"Test acc={tm['acc']:.3f} bal_acc={tm['bal_acc']:.3f} auc={tm['auc']:.3f} ap={tm['ap']:.3f} "
            f"spec={tm.get('specificity', float('nan')):.3f} fpr={tm.get('false_positive_rate', float('nan')):.3f}"
        )
        test_per_source = metrics_per_source(test_df, ty, tp, float(best_thr), min_samples=5)
        if test_per_source:
            parts = [
                f"{k}:n={v.get('n', '')},f1={float(v.get('f1', 0.0)):.2f},spec={float(v.get('specificity', 0.0)):.2f},fpr={float(v.get('false_positive_rate', 0.0)):.2f}"
                for k, v in list(test_per_source.items())[:10]
            ]
            tail = " ..." if len(test_per_source) > 10 else ""
            print("[test] per_source (shared thr):", " | ".join(parts) + tail)

        extra_y_neg_sv = extra_p_neg_sv = None
        if extra_test_loader is not None:
            ey, ep_probs = eval_probs(model, extra_test_loader, device, temperature=T)
            pred = (ep_probs >= float(best_thr)).astype(np.int64)
            ey_i = np.asarray(ey, dtype=np.int64)
            neg_m = ey_i == 0
            n_neg = int(neg_m.sum())
            pred_neg = pred[neg_m]
            n_fp = int((pred_neg == 1).sum())
            n_tn = int((pred_neg == 0).sum())
            fp_rate = n_fp / max(1, n_neg)
            print(
                f"Extra test (no_fire subset @ thr={best_thr:.3f}) "
                f"n_neg={n_neg} FP={n_fp} TN={n_tn} FPR_neg={fp_rate:.3f} "
                f"(full n={len(ey_i)})"
            )
            if n_neg > 0:
                extra_y_neg_sv = ey_i[neg_m]
                extra_p_neg_sv = np.asarray(ep_probs, dtype=np.float64)[neg_m]

        if scheduler_kind == "plateau":
            sched.step(vm["ap"] if vm["ap"] == vm["ap"] else 0.0)
        else:
            sched.step()

        score_legacy = 0.5 * float(vm.get("f1", 0.0)) + 0.5 * float(vm.get("bal_acc", 0.0))
        score_realistic = realistic_selection_score(vm)
        selection_score = score_realistic if sel_norm == "realistic" else score_legacy

        # checkpoint selection metric: legacy = 0.5*F1 + 0.5*BalAcc;
        # realistic = F1 + BalAcc + AP - 0.5*FPR (val @ best_thr)
        if selection_score == selection_score and selection_score > best_val_ap:
            best_val_ap = float(selection_score)
            patience_counter = 0
            thr_alarm_raw = _best_threshold_mode(vy, vp, "alarm")
            # Precision-biased (review) threshold is never clamped — keep strict.
            thr_review = _best_threshold_mode(vy, vp, "review")
            # Clamp the alarm threshold from below so the production default is
            # never dangerously permissive. We store both the raw and clamped
            # values for transparency / sweep reports.
            thr_alarm = max(float(THRESHOLD_ALARM_MIN), float(thr_alarm_raw))
            if thr_alarm != thr_alarm_raw:
                print(
                    f"[val] threshold_alarm clamped {thr_alarm_raw:.3f} -> {thr_alarm:.3f} "
                    f"(min={THRESHOLD_ALARM_MIN})"
                )
            # Worst-source diagnostics for the current best epoch.
            worst_source_by_fpr = None
            worst_source_by_recall = None
            if per_source:
                try:
                    worst_source_by_fpr = max(
                        per_source.items(),
                        key=lambda kv: float(kv[1].get("false_positive_rate", 0.0) or 0.0),
                    )[0]
                    worst_source_by_recall = min(
                        per_source.items(),
                        key=lambda kv: float(kv[1].get("recall", 1.0) or 0.0),
                    )[0]
                except Exception:
                    pass
            threshold_policy_csv = ""
            threshold_policies: dict = {}
            video_event_metrics = None
            try:
                pol_grid = threshold_sweep_grid(
                    vy,
                    vp,
                    ty,
                    tp,
                    thresholds=np.arange(0.10, 0.96, 0.05),
                    extra_y_neg=extra_y_neg_sv,
                    extra_p_neg=extra_p_neg_sv,
                )
                policy_path = Path(OUTPUTS_DIR) / f"threshold_policy_{mode}_{mf}.csv"
                policy_path.parent.mkdir(parents=True, exist_ok=True)
                pol_grid.to_csv(policy_path, index=False)
                threshold_policy_csv = str(policy_path)
                threshold_policies = select_threshold_policies(pol_grid)
                print(f"[eval] threshold_policy_csv -> {threshold_policy_csv}")
            except Exception as e:
                print(f"[eval] threshold_policy sweep skipped: {type(e).__name__}: {e}")
            try:
                seq = compute_sequence_alarm_summary(
                    test_df,
                    ty,
                    tp,
                    prob_threshold_high=float(best_thr),
                    prob_threshold_low=None,
                )
                video_event_metrics = sanitize_for_json(seq) if seq is not None else None
                if video_event_metrics is not None and not video_event_metrics.get("skipped"):
                    print(
                        f"[eval] seq alarm: FA_events={video_event_metrics.get('false_alarm_event_count')} "
                        f"missed={video_event_metrics.get('missed_fire_event_count')} "
                        f"lat_mean={video_event_metrics.get('detection_latency_frames_mean')}"
                    )
                elif isinstance(video_event_metrics, dict) and video_event_metrics.get("skipped"):
                    print(f"[eval] seq alarm skipped: {video_event_metrics.get('reason')}")
            except Exception as e:
                print(f"[eval] seq alarm skipped: {type(e).__name__}: {e}")
                video_event_metrics = {"skipped": True, "reason": str(e)}

            torch.save(
                {
                    "mode": mode,
                    "model_family": mf,
                    "in_ch": in_ch,
                    "backbone": backbone,
                    "input_size": int(size),
                    "class_mapping": {"0": "no_fire", "1": "fire"},
                    "training_args": {
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
                    },
                    "state": model.state_dict(),
                    # Default inference threshold (operating point) and analysis threshold (F1-optimal)
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
                    "temperature": float(T),
                    "worst_source_by_fpr": worst_source_by_fpr,
                    "worst_source_by_recall": worst_source_by_recall,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                out_ckpt,
            )
            print(
                f"Saved (best sel={sel_norm} selection_score={selection_score:.4f}, "
                f"f1+balacc_legacy={score_legacy:.4f}, realistic={score_realistic:.4f}, T={T:.3f}): {out_ckpt}"
            )
            if worst_source_by_fpr or worst_source_by_recall:
                print(
                    f"[val] worst_source_by_fpr={worst_source_by_fpr} "
                    f"worst_source_by_recall={worst_source_by_recall}"
                )
            extra_info = {}
            fp_xt_written = Path(OUTPUTS_DIR) / f"extra_test_false_positives_{mode}_{mf}.csv"
            if extra_test_loader is not None:
                ey_i = np.asarray(ey, dtype=np.int64)
                ep_i = np.asarray(ep_probs, dtype=np.float64)
                pred_i = (ep_i >= float(best_thr)).astype(np.int64)
                nm_ck = ey_i == 0
                n_neg_ck = int(nm_ck.sum())
                pred_neg_ck = pred_i[nm_ck]
                fp_n_ck = int((pred_neg_ck == 1).sum())
                tn_n_ck = int((pred_neg_ck == 0).sum())
                fpr_n_ck = float(fp_n_ck / max(1, n_neg_ck))
                extra_info = {
                    "extra_test_n": len(ey_i),
                    "extra_test_n_no_fire": n_neg_ck,
                    "extra_test_fp_no_fire": fp_n_ck,
                    "extra_test_tn_no_fire": tn_n_ck,
                    "extra_test_false_positive_rate_no_fire": fpr_n_ck,
                    # Backward-compat keys (subset semantics = no_fire / negatives)
                    "extra_test_fp": fp_n_ck,
                    "extra_test_tn": tn_n_ck,
                    "extra_test_fp_rate": fpr_n_ck,
                }
                fp_mask_xt = nm_ck & (pred_i == 1)
                cols_xt = [
                    "path_rgb",
                    "path_th",
                    "source",
                    "split",
                    "key",
                    "split_group",
                    "label",
                    "prob_fire",
                    "threshold",
                    "pred",
                ]
                if len(extra_test_df) != len(ey_i):
                    pd.DataFrame(columns=cols_xt).to_csv(fp_xt_written, index=False)
                    print(
                        "[extra_test][WARN] extra_test row count != predictions; wrote empty FP template",
                        flush=True,
                    )
                    print(f"[extra_test] false_positives saved: {fp_xt_written} (n=0)", flush=True)
                elif fp_mask_xt.any():
                    xt_base = extra_test_df.iloc[np.flatnonzero(fp_mask_xt)].copy()
                    xt_base.loc[:, "prob_fire"] = ep_i[fp_mask_xt].astype(np.float32)
                    xt_base.loc[:, "threshold"] = float(best_thr)
                    xt_base.loc[:, "pred"] = pred_i[fp_mask_xt]
                    present_xt = [c for c in cols_xt if c in xt_base.columns]
                    xt_base[present_xt].to_csv(fp_xt_written, index=False)
                    print(
                        f"[extra_test] false_positives saved: {fp_xt_written} "
                        f"(n={len(xt_base)})",
                        flush=True,
                    )
                else:
                    pd.DataFrame(columns=cols_xt).to_csv(fp_xt_written, index=False)
                    print(f"[extra_test] false_positives saved: {fp_xt_written} (n=0)", flush=True)

            metrics = {
                "mode": mode,
                "model_family": mf,
                "epoch": ep,
                "threshold": float(best_thr),
                "threshold_alarm": float(thr_alarm),
                "threshold_alarm_raw": float(thr_alarm_raw),
                "threshold_alarm_clamped": float(thr_alarm),
                "threshold_alarm_min": float(THRESHOLD_ALARM_MIN),
                "threshold_review": float(thr_review),
                "temperature": float(T),
                "ece_val": float(ece),
                "brier_val": float(brier),
                "reliability_val": reliability_report(vy, vp, n_bins=10),
                "val_score_f1_balacc": float(score_legacy),
                "val_score_realistic": float(score_realistic),
                "val_selection_score": float(selection_score),
                "selection_metric": str(sel_norm),
                "training_class_balance": training_class_balance,
                "val_per_source": per_source,
                "test_per_source": test_per_source,
                "gap_metrics": {
                    "fpr_gap": (
                        float(vm.get("false_positive_rate", float("nan")))
                        - float(tm.get("false_positive_rate", float("nan")))
                    ),
                    "recall_gap": (
                        float(vm.get("recall", float("nan")))
                        - float(tm.get("recall", float("nan")))
                    ),
                    "f1_gap": (
                        float(vm.get("f1", float("nan")))
                        - float(tm.get("f1", float("nan")))
                    ),
                    "bal_acc_gap": (
                        float(vm.get("bal_acc", float("nan")))
                        - float(tm.get("bal_acc", float("nan")))
                    ),
                    "ece_gap": float(ece),  # val ECE; test ECE captured per-source
                },
                "threshold_policies": threshold_policies,
                "threshold_policy_csv": threshold_policy_csv or None,
                "video_event_metrics": video_event_metrics if video_event_metrics is not None else {},
                "worst_source_by_fpr": worst_source_by_fpr,
                "worst_source_by_recall": worst_source_by_recall,
                "val": {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in vm.items()},
                "test": {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in tm.items()},
                **extra_info,
            }
            if hard_negative_csv and Path(str(hard_negative_csv)).exists():
                metrics["hard_negative_training_stats"] = dict(_hn_dbg)
                metrics["hard_negative_csv"] = str(hard_negative_csv)
                metrics["hard_negative_weight"] = float(hard_negative_weight)
            try:
                _aug_dbg = dict(getattr(train_ds, "aug", {}) or {})
            except Exception:
                _aug_dbg = {}
            metrics["augmentation"] = {
                "profile": aug_str_norm,
                "modal_dropout_p": float(modal_dropout_p),
                "params": {
                    k: _aug_dbg.get(k)
                    for k in (
                        "p_jitter",
                        "p_blur",
                        "blur_radius_max",
                        "p_rgb_noise",
                        "sigma_rgb",
                        "p_thermal_noise",
                        "sigma_thermal",
                        "p_thermal_shift_scale",
                        "thermal_scale_jitter",
                        "thermal_shift_jitter",
                        "p_combined_noise",
                        "p_random_erase",
                    )
                },
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
                print(f"Early stop (val AP did not improve for {patience} epochs)")
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
            "best_val_score": float(best_val_ap),
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
        ws_fpr = ck.get("worst_source_by_fpr")
        ws_rec = ck.get("worst_source_by_recall")
        if ws_fpr or ws_rec:
            print(
                f"[calibrate] worst_source_by_fpr={ws_fpr} worst_source_by_recall={ws_rec}"
            )

    # Final report: compare fixed thresholds on val/test using the best checkpoint.
    try:
        if Path(out_ckpt).exists():
            try:
                ck = torch.load(out_ckpt, map_location=device, weights_only=True)
            except TypeError:
                ck = torch.load(out_ckpt, map_location=device)
            model.load_state_dict(ck["state"])
            model.eval()
            T_best = float(ck.get("temperature", 1.0))
            fixed = [0.50, 0.55]
            vy2, vp2 = eval_probs(model, val_loader, device, temperature=T_best)
            ty2, tp2 = eval_probs(model, test_loader, device, temperature=T_best)
            print("[final] threshold comparison (val/test):")
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

    if robustness_eval and mode == "fusion" and Path(out_ckpt).exists():
        metrics_rb = Path(OUTPUTS_DIR) / f"metrics_{mode}_{mf}.json"
        if mf == "early_fusion" and mode == "fusion":
            metrics_rb = Path(OUTPUTS_DIR) / f"metrics_{mode}.json"
        hn_csv_on = bool(hard_negative_csv and Path(str(hard_negative_csv)).is_file())
        if hn_csv_on and len(extra_test_df) == 0:
            print(
                "[robustness_eval] skipped: extra_test holdout has no rows after excluding hard-negative training overlaps",
                flush=True,
            )
        else:
            flame_rb = flame3_eval_slice(test_df, extra_test_df)
            flame_rb, drop_rb = filter_df_existing_paths(flame_rb, mode="fusion")
            flame_rb = flame_rb.reset_index(drop=True)
            if drop_rb:
                print(f"[robustness] dropped {drop_rb} FLAME3 eval rows (missing fusion files)")
            if len(flame_rb) == 0:
                msg_tail = ""
                if hn_excluded_xt and len(extra_test_df) == 0:
                    msg_tail = " — extra_test was fully excluded for unbiased evaluation"
                print(
                    f"[robustness] skipped: no flame3 / flame3_raw_extra rows in test ∪ extra_test after path filter{msg_tail}"
                )
            else:
                try:
                    try:
                        ck_rb = torch.load(out_ckpt, map_location=device, weights_only=True)
                    except TypeError:
                        ck_rb = torch.load(out_ckpt, map_location=device)
                    model.load_state_dict(ck_rb["state"])
                    model.eval()
                    thr_rb = float(ck_rb.get("threshold", 0.5))
                    temp_rb = float(ck_rb.get("temperature", 1.0))
                    rb_block = run_flame3_robustness_evaluation(
                        model,
                        device,
                        flame3_eval_df=flame_rb,
                        temperature=temp_rb,
                        threshold=thr_rb,
                        batch_size=int(bs),
                        size=int(size),
                        thermal_norm=thermal_norm,
                        num_workers=int(num_workers),
                        pin_memory=bool(loader_common.get("pin_memory", False)),
                    )
                    if rb_block:
                        merge_robustness_into_metrics_json(metrics_rb, rb_block)
                        print(
                            f"[robustness] wrote robustness_eval ({len(flame_rb)} rows, thr={thr_rb:.4f}, T={temp_rb:.4f}) → {metrics_rb}"
                        )
                        try:
                            augment_metrics_json_with_robustness_outputs(
                                metrics_rb, csv_tag=f"{mode}_{mf}"
                            )
                        except Exception as e:
                            print(f"[robustness] summary augmentation skipped: {type(e).__name__}: {e}")
                except Exception as e:
                    print(f"[robustness] skipped: {type(e).__name__}: {e}")

    return out_ckpt
