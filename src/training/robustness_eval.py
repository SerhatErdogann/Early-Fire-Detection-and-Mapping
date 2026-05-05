"""
Evaluation-time corruption for FLAME3 (fusion) subsets only — never used during training.

Corruptions apply to tensors already normalized like ``FlameDataset`` fusion batches:
shape ``[B, 4, H, W]`` with RGB in channels ``0:3`` and thermal in ``3:4``, values in ``[0, 1]``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from torchvision.transforms.functional import gaussian_blur
except ImportError:
    gaussian_blur = None  # pragma: no cover

from ..data.dataset import FlameDataset
from .eval_reporting import _jsonable_metric_row, metrics_per_source, sanitize_for_json
from .metrics import brier_score_binary, eval_probs, expected_calibration_error, metrics_at_threshold


ROBUSTNESS_VARIANTS: tuple[str, ...] = (
    "clean",
    "rgb_gaussian_noise",
    "rgb_brightness_contrast",
    "rgb_blur",
    "thermal_gaussian_noise",
    "thermal_shift_scale",
    "rgb_thermal_combined_noise",
)

_DEVICE_RNG_FALLBACK_WARNED = False


def package_robustness_metrics(m: dict) -> dict[str, Any]:
    """Scalar JSON-friendly metrics incl. ``fpr`` alias."""
    row = dict(_jsonable_metric_row(m))
    row["fpr"] = float(row.get("false_positive_rate", float("nan")))
    return row


def _rng_seed(base_seed: int, variant_id: int, batch_index: int, salt: int = 0) -> int:
    """Deterministic seed; fits ``torch.Generator.manual_seed`` int range."""
    s = int(base_seed) + int(variant_id) * 9176 + int(batch_index) * 31337 + int(salt) * 7937
    return int(s % (2**63 - 1))


def _standard_normal_noise(
    like: torch.Tensor,
    *,
    base_seed: int,
    variant_id: int,
    batch_index: int,
    salt: int = 0,
) -> torch.Tensor:
    """Normal(0,1) tensor matching ``like.shape``, ``dtype``, and ``like.device``.

    Uses ``torch.Generator(device=like.device)`` when the build supports it on the
    accelerator; otherwise CPU generator + ``.to(like.device)`` so RNG never pairs
    a CPU ``Generator`` with accelerator ``torch.randn``.
    """
    global _DEVICE_RNG_FALLBACK_WARNED
    dev = like.device
    dtype = like.dtype
    shape = tuple(like.shape)
    seed = _rng_seed(base_seed, variant_id, batch_index, salt)

    if dev.type != "cpu":
        try:
            gen = torch.Generator(device=dev)
        except (TypeError, RuntimeError):
            gen = torch.Generator(device=torch.device("cpu"))
            gen.manual_seed(seed)
            if not _DEVICE_RNG_FALLBACK_WARNED:
                print(
                    "[robustness][WARN] torch.Generator(device=accelerator) not available; "
                    "using CPU RNG + .to(device) for noise (deterministic seed preserved).",
                    flush=True,
                )
                _DEVICE_RNG_FALLBACK_WARNED = True
            out = torch.randn(shape, dtype=dtype, device=torch.device("cpu"), generator=gen).to(dev, non_blocking=True)
        else:
            gen.manual_seed(seed)
            out = torch.randn(shape, dtype=dtype, device=dev, generator=gen)
    else:
        gen = torch.Generator(device=torch.device("cpu"))
        gen.manual_seed(seed)
        out = torch.randn(shape, dtype=dtype, device=dev, generator=gen)

    if out.device != dev:
        raise RuntimeError(f"[robustness] noise device mismatch: expected {dev}, got {out.device}")
    return out


class FusionBatchCorrupter:
    """Stateful closure for ``eval_probs(..., corrupt_batch=...)``. Deterministic batch order."""

    __slots__ = ("variant", "base_seed", "variant_id", "counter")

    def __init__(self, variant: str, *, variant_id: int = 0, base_seed: int = 928_443) -> None:
        self.variant = (variant or "clean").strip().lower()
        self.base_seed = int(base_seed)
        self.variant_id = int(variant_id)
        self.counter = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        v = self.variant
        if v == "clean":
            return x
        bidx = self.counter
        if bidx == 0:
            print(f"[robustness] variant={self.variant} device={x.device} shape={tuple(x.shape)}", flush=True)
        self.counter += 1
        vd = self.variant_id

        rgb = x[:, :3].clone()
        th = x[:, 3:4].clone()

        out: torch.Tensor
        if v == "rgb_gaussian_noise":
            noise = _standard_normal_noise(rgb, base_seed=self.base_seed, variant_id=vd, batch_index=bidx, salt=1)
            rgb = (rgb + 0.038 * noise).clamp(0.0, 1.0)
            out = torch.cat([rgb, th], dim=1)

        elif v == "rgb_brightness_contrast":
            rgb = (rgb * 1.10 + 0.025).clamp(0.0, 1.0)
            rgb = (((rgb - 0.5) * 1.08) + 0.5).clamp(0.0, 1.0)
            out = torch.cat([rgb, th], dim=1)

        elif v == "rgb_blur":
            if gaussian_blur is None:
                raise RuntimeError(
                    "[robustness] variant rgb_blur requires torchvision (torchvision.transforms.functional.gaussian_blur). "
                    "Install torchvision or remove rgb_blur from ROBUSTNESS_VARIANTS."
                )
            out_rgb = gaussian_blur(rgb, kernel_size=[5, 5], sigma=[1.6, 1.6])
            out = torch.cat([out_rgb, th], dim=1)

        elif v == "thermal_gaussian_noise":
            noise = _standard_normal_noise(th, base_seed=self.base_seed, variant_id=vd, batch_index=bidx, salt=2)
            th = (th + 0.042 * noise).clamp(0.0, 1.0)
            out = torch.cat([rgb, th], dim=1)

        elif v == "thermal_shift_scale":
            th = ((th - 0.5) * 1.14 + 0.52).clamp(0.0, 1.0)
            out = torch.cat([rgb, th], dim=1)

        elif v == "rgb_thermal_combined_noise":
            nr = _standard_normal_noise(rgb, base_seed=self.base_seed, variant_id=vd, batch_index=bidx, salt=3)
            nt = _standard_normal_noise(th, base_seed=self.base_seed, variant_id=vd, batch_index=bidx, salt=4)
            rgb = (rgb + 0.028 * nr).clamp(0.0, 1.0)
            rgb = (rgb * 1.06 + 0.015).clamp(0.0, 1.0)
            th = (th + 0.035 * nt).clamp(0.0, 1.0)
            out = torch.cat([rgb, th], dim=1)

        else:
            raise ValueError(
                f"[robustness] unknown variant {self.variant!r}. Expected one of {ROBUSTNESS_VARIANTS}"
            )

        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
        if out.device != x.device:
            raise RuntimeError(f"[robustness] corruption output device {out.device} != input {x.device}")
        return out


_SOURCES_FLAME3_ROBUSTNESS = frozenset({"flame3", "flame3_raw_extra"})


def flame3_eval_slice(test_df: pd.DataFrame, extra_test_df: pd.DataFrame) -> pd.DataFrame:
    """FLAME-style paired rows for robustness: official ``flame3`` test + Raw ``extra_test`` holdout."""
    parts: list[pd.DataFrame] = []
    if test_df is not None and len(test_df):
        m = test_df["source"].astype(str).isin(_SOURCES_FLAME3_ROBUSTNESS)
        parts.append(test_df.loc[m].copy())
    if extra_test_df is not None and len(extra_test_df):
        m = extra_test_df["source"].astype(str).isin(_SOURCES_FLAME3_ROBUSTNESS)
        parts.append(extra_test_df.loc[m].copy())
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_flame3_robustness_evaluation(
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    flame3_eval_df: pd.DataFrame,
    temperature: float,
    threshold: float,
    variant_names: tuple[str, ...] = ROBUSTNESS_VARIANTS,
    batch_size: int = 16,
    size: int = 384,
    thermal_norm: str = "percentile",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, Any]:
    """
    Evaluate fusion model on FLAME3 rows under corruption variants only at inference time.

    Uses caller-provided **temperature** and **threshold** (clean val strategy).
    """
    df = flame3_eval_df.reset_index(drop=True).copy()
    if df.empty:
        return {}

    loaders_common = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        drop_last=False,
    )

    block: dict[str, Any] = {
        "meta": {
            "subset": "flame3 (+ flame3_raw_extra) in test ∪ extra_test",
            "n_rows": int(len(df)),
            "threshold": float(threshold),
            "temperature": float(temperature),
            "variants": list(variant_names),
        },
    }

    for i, vn in enumerate(variant_names):
        try:
            mdev = next(model.parameters()).device
        except StopIteration:
            mdev = torch.device(device)
        if str(vn) == "clean":
            print(
                f"[robustness] variant=clean device={mdev} shape=(batch,4,{int(size)},{int(size)})",
                flush=True,
            )
        ds = FlameDataset(df, mode="fusion", size=int(size), train=False, thermal_norm=str(thermal_norm))
        dl = DataLoader(ds, **loaders_common)
        corrupt_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        if str(vn) != "clean":
            corrupt_fn = FusionBatchCorrupter(str(vn), variant_id=i)
        ys, ps = eval_probs(model, dl, device, temperature=float(temperature), corrupt_batch=corrupt_fn)

        mt = metrics_at_threshold(ys, ps, float(threshold))
        flat = package_robustness_metrics(mt)
        flat.pop("cm", None)
        cm = mt.get("cm")

        pkg: dict[str, Any] = {
            **flat,
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
            "ece": float(expected_calibration_error(ys, ps)),
            "brier": float(brier_score_binary(ys, ps)),
            "metrics_per_source": {},
        }
        ps_src = metrics_per_source(df, ys, ps, float(threshold), min_samples=1)
        for src, row_d in ps_src.items():
            d2 = {k: v for k, v in row_d.items() if k != "cm"}
            cm_s = row_d.get("cm")
            if hasattr(cm_s, "tolist"):
                d2["confusion_matrix"] = cm_s.tolist()
            # Short fpr alias for nested per-source
            if "false_positive_rate" in d2:
                d2["fpr"] = float(d2["false_positive_rate"])
            pkg["metrics_per_source"][src] = d2

        block[str(vn)] = pkg

    return block


def merge_robustness_into_metrics_json(metrics_path: Path, robustness_block: dict[str, Any]) -> None:
    """Merge variant metrics under ``robustness_eval``; settings under ``robustness_eval_meta``."""
    mp = Path(metrics_path)
    payload: dict[str, Any]
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = {}
    meta = robustness_block.get("meta")
    variants_only = {k: v for k, v in robustness_block.items() if k != "meta"}
    payload["robustness_eval"] = variants_only
    if meta is not None:
        payload["robustness_eval_meta"] = meta

    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(payload), fh, indent=2, ensure_ascii=False)


def _variant_fpr(pkg: dict[str, Any]) -> float:
    v = pkg.get("fpr", pkg.get("false_positive_rate", float("nan")))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_robustness_summary_payload(variants: dict[str, Any]) -> dict[str, Any] | None:
    """Compress ``robustness_eval`` variants into headline FPR deltas for metrics JSON."""
    if not variants:
        return None
    get = variants.get
    cf = _variant_fpr(get("clean", {}) or {})

    rg = [_variant_fpr(get(v, {}) or {}) for v in ("rgb_gaussian_noise", "rgb_brightness_contrast", "rgb_blur")]
    tg = [_variant_fpr(get(v, {}) or {}) for v in ("thermal_gaussian_noise", "thermal_shift_scale")]
    rg_ok = [x for x in rg if x == x]
    tg_ok = [x for x in tg if x == x]
    rgb_mu = float(sum(rg_ok) / len(rg_ok)) if rg_ok else float("nan")
    th_mu = float(sum(tg_ok) / len(tg_ok)) if tg_ok else float("nan")
    cob = _variant_fpr(get("rgb_thermal_combined_noise", {}) or {})

    worst_v, worst_f = "", float("-inf")
    for name in ROBUSTNESS_VARIANTS:
        if name == "clean":
            continue
        fp = _variant_fpr(get(name, {}) or {})
        if fp == fp and fp >= worst_f:
            worst_f, worst_v = fp, name

    inc = cob - cf if (cob == cob and cf == cf) else float("nan")
    out = {
        "clean_fpr": cf,
        "rgb_noise_fpr": rgb_mu,
        "thermal_noise_fpr": th_mu,
        "combined_noise_fpr": cob,
        "worst_variant": worst_v,
        "worst_variant_fpr": worst_f if worst_v else float("nan"),
        "fpr_increase_combined_vs_clean": inc,
    }
    return sanitize_for_json(out)


def augment_metrics_json_with_robustness_outputs(metrics_path: Path, *, csv_tag: str) -> None:
    """Adds ``robustness_summary`` to metrics JSON root and writes ``robustness_summary_{csv_tag}.csv``."""
    try:
        from config import OUTPUTS_DIR as _OUT
    except ImportError:
        _OUT = Path("outputs")

    mp = Path(metrics_path)
    if not mp.exists():
        return
    try:
        with open(mp, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return

    variants = payload.get("robustness_eval")
    summary = build_robustness_summary_payload(variants or {})
    if summary:
        payload["robustness_summary"] = summary

    rows_out: list[dict[str, Any]] = []
    if isinstance(variants, dict):
        for vn in ROBUSTNESS_VARIANTS:
            pkg = variants.get(vn)
            if not isinstance(pkg, dict):
                continue
            row_d: dict[str, Any] = {"variant": vn}
            keys = (
                ("accuracy", "acc"),
                ("balanced_accuracy", "bal_acc"),
                ("precision", "precision"),
                ("recall", "recall"),
                ("specificity", "specificity"),
                ("fpr", "fpr"),
                ("f1", "f1"),
                ("auc", "auc"),
                ("ap", "ap"),
                ("ece", "ece"),
                ("brier", "brier"),
            )
            for col, kk in keys:
                row_d[col] = pkg.get(kk, float("nan"))
            rows_out.append(row_d)

    out_dir = Path(_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / f"robustness_summary_{csv_tag}.csv"
    pd.DataFrame(rows_out).to_csv(csv_p, index=False)

    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(payload), fh, indent=2, ensure_ascii=False)

    print(f"[robustness] summary appended to {mp}; CSV -> {csv_p}", flush=True)
