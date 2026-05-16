"""Offline corruption evaluation for trained checkpoints.

Default CLI matches training protocol: ``gauss_noise_rgb`` @ severity **1**
(RGB/fusion); thermal-only checkpoints use ``gauss_noise_thermal`` @ severity **1**.
Corruptions run **only** during this eval forward path (not in training/inference UI).

Optional ``--legacy_grid`` restores the wider (corruption × severity) sweep incl. ``clean``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FlameDataset  # noqa: E402
from src.data.path_filter import filter_df_existing_paths  # noqa: E402
from src.eval.thermal_calibration import (  # noqa: E402
    resolve_thermal_calibration_or_exit,
    thermal_norm_from_checkpoint,
)
from src.models import make_classifier  # noqa: E402
from src.training.metrics import metrics_at_threshold  # noqa: E402


# ---------------------------------------------------------------------------
# Corruption transforms
# ---------------------------------------------------------------------------
# Tensors are expected as (B, C, H, W) float32 with the convention used by
# ``FlameDataset``: channels 0:3 = RGB scaled to [0, 1]; channel 3 (when
# present, i.e. fusion / thermal modes) = thermal scaled to [0, 1].

CorruptionFn = Callable[[torch.Tensor, int], torch.Tensor]

_RGB_CHANNELS = slice(0, 3)
_THERMAL_CHANNELS = slice(3, None)


def _has_thermal(x: torch.Tensor) -> bool:
    return x.dim() == 4 and x.shape[1] >= 4


def _gauss_noise_rgb(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma_table = {1: 0.02, 2: 0.05, 3: 0.10}
    sigma = sigma_table.get(int(severity), 0.05)
    if x.shape[1] < 1:
        return x
    out = x.clone()
    noise = torch.randn_like(out[:, _RGB_CHANNELS]) * sigma
    out[:, _RGB_CHANNELS] = (out[:, _RGB_CHANNELS] + noise).clamp_(0.0, 1.0)
    return out


def _gauss_noise_thermal(x: torch.Tensor, severity: int) -> torch.Tensor:
    if not _has_thermal(x):
        return x
    sigma_table = {1: 0.02, 2: 0.05, 3: 0.10}
    sigma = sigma_table.get(int(severity), 0.05)
    out = x.clone()
    noise = torch.randn_like(out[:, _THERMAL_CHANNELS]) * sigma
    out[:, _THERMAL_CHANNELS] = (out[:, _THERMAL_CHANNELS] + noise).clamp_(0.0, 1.0)
    return out


def _brightness_contrast(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Apply a deterministic brightness + contrast shift on RGB only."""
    delta_table = {1: 0.10, 2: 0.20, 3: 0.30}
    delta = delta_table.get(int(severity), 0.20)
    contrast_table = {1: 1.10, 2: 1.20, 3: 1.30}
    contrast = contrast_table.get(int(severity), 1.20)
    out = x.clone()
    rgb = out[:, _RGB_CHANNELS]
    mean = rgb.mean(dim=(2, 3), keepdim=True)
    rgb = (rgb - mean) * contrast + mean + delta
    out[:, _RGB_CHANNELS] = rgb.clamp_(0.0, 1.0)
    return out


def _gaussian_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma_table = {1: 0.5, 2: 1.0, 3: 2.0}
    sigma = float(sigma_table.get(int(severity), 1.0))
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    coords = torch.arange(ksize, dtype=torch.float32, device=x.device) - half
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kernel_2d = g1d.view(-1, 1) @ g1d.view(1, -1)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=half, groups=c)


def _thermal_shift(x: torch.Tensor, severity: int) -> torch.Tensor:
    if not _has_thermal(x):
        return x
    shift_table = {1: 0.05, 2: 0.10, 3: 0.20}
    shift = float(shift_table.get(int(severity), 0.10))
    out = x.clone()
    out[:, _THERMAL_CHANNELS] = (out[:, _THERMAL_CHANNELS] + shift).clamp_(0.0, 1.0)
    return out


CORRUPTIONS: dict[str, CorruptionFn] = {
    "gauss_noise_rgb": _gauss_noise_rgb,
    "gauss_noise_thermal": _gauss_noise_thermal,
    "brightness_contrast": _brightness_contrast,
    "gaussian_blur": _gaussian_blur,
    "thermal_shift": _thermal_shift,
}

def protocol_corruption(mode: str) -> tuple[str, int]:
    """Single operational eval corruption (name, severity), matching the training protocol."""
    if str(mode).lower() == "thermal":
        return ("gauss_noise_thermal", 1)
    return ("gauss_noise_rgb", 1)


def _resolve_corruptions(spec: str, mode: str) -> list[str]:
    """Expand the CLI ``--corruptions`` flag into a concrete list of names."""
    spec = (spec or "all").strip()
    if spec.lower() in {"all", "*"}:
        names = list(CORRUPTIONS.keys())
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
        unknown = [n for n in names if n not in CORRUPTIONS]
        if unknown:
            raise SystemExit(
                f"Unknown corruption name(s): {unknown}. "
                f"Available: {sorted(CORRUPTIONS.keys())}"
            )
    if mode == "rgb":
        names = [n for n in names if not n.startswith("gauss_noise_thermal") and n != "thermal_shift"]
    elif mode == "thermal":
        names = [n for n in names if n not in ("gauss_noise_rgb", "brightness_contrast")]
    return names


def realistic_noisy_corruption_names(mode: str) -> list[str]:
    """Corruption name(s) active for protocol eval on ``mode``."""
    cn, _ = protocol_corruption(mode)
    resolved = _resolve_corruptions(str(cn), str(mode).lower())
    return [n for n in resolved if n == cn]


# ---------------------------------------------------------------------------
# Checkpoint / data plumbing
# ---------------------------------------------------------------------------


def _load_index(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the rows belonging to ``split`` ('val' / 'test' / 'all')."""
    split = (split or "test").lower()
    if split == "all":
        return df
    if "split" in df.columns:
        sp = df["split"].astype(str).str.lower()
        sub = df[sp == split].copy()
        if len(sub):
            return sub
        print(f"[robustness] WARN: split column has no '{split}' rows; using all rows.")
        return df
    print(f"[robustness] WARN: index has no 'split' column; using all rows for split={split!r}.")
    return df


def _load_checkpoint(ckpt_path: str, device: str) -> dict:
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    return ck


def _build_model(ck: dict, device: str) -> tuple[torch.nn.Module, dict]:
    mode = str(ck.get("mode") or "fusion").lower()
    family = str(ck.get("model_family") or "dual_branch_fusion").lower()
    backbone = str(ck.get("backbone") or "resnet50")
    size = int(ck.get("input_size") or 384)
    model = make_classifier(family, backbone, mode, num_classes=2, pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    info = {
        "mode": mode,
        "family": family,
        "backbone": backbone,
        "size": size,
        "threshold": float(ck.get("threshold", 0.5)),
        "temperature": float(ck.get("temperature", 1.0)),
    }
    return model, info


# ---------------------------------------------------------------------------
# Forward pass + metric aggregation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float,
    corruption: CorruptionFn | None,
    severity: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys: list[int] = []
    probs: list[float] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False)
        if corruption is not None:
            xb = corruption(xb, severity)
        logits = model(xb)
        scaled = logits / max(1e-6, float(temperature))
        p_fire = torch.softmax(scaled, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(p_fire.tolist())
        ys.extend(yb.detach().cpu().numpy().astype(int).tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def _row_for(ys: np.ndarray, probs: np.ndarray, threshold: float, name: str, severity: int | str) -> dict:
    if len(ys) == 0:
        return {
            "corruption": name,
            "severity": severity,
            "n": 0,
            "error": "empty_split",
        }
    m = metrics_at_threshold(ys, probs, float(threshold))
    return {
        "corruption": name,
        "severity": severity,
        "n": int(len(ys)),
        "acc": float(m.get("acc", float("nan"))),
        "bal_acc": float(m.get("bal_acc", float("nan"))),
        "precision": float(m.get("precision", float("nan"))),
        "recall": float(m.get("recall", float("nan"))),
        "f1": float(m.get("f1", float("nan"))),
        "specificity": float(m.get("specificity", float("nan"))),
        "false_positive_rate": float(m.get("false_positive_rate", float("nan"))),
        "auc": float(m.get("auc", float("nan"))),
        "ap": float(m.get("ap", float("nan"))),
        "threshold": float(threshold),
    }


@torch.inference_mode()
def eval_logits_corrupted(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    corruption_name: str,
    severity: int,
    seed: int = 0,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass logits with corruption on inputs (temperature applied later by trainer)."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    fn = CORRUPTIONS[str(corruption_name)]
    ys: list[int] = []
    logits_list: list[np.ndarray] = []
    bi = 0
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False)
        xb = fn(xb, int(severity))
        with torch.no_grad():
            logits = model(xb)
        logits_list.append(logits.detach().cpu().numpy())
        ys.extend(yb.detach().cpu().numpy().astype(int).tolist())
        bi += 1
        if max_batches is not None and bi >= int(max_batches):
            break
    if not logits_list:
        return np.asarray([], dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    return np.asarray(ys, dtype=np.int64), np.concatenate(logits_list, axis=0)


def _parse_severities(spec: str) -> list[int]:
    out: list[int] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if 1 <= v <= 3:
            out.append(v)
    return out or [1]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_robustness(
    ckpt_path: str,
    csv_path: str,
    split: str = "test",
    corruptions: Iterable[str] | str = "all",
    severities: Iterable[int] | str = (1,),
    out_csv: str | None = "outputs/robustness_eval.csv",
    bs: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    temperature_override: float | None = None,
    threshold_override: float | None = None,
    seed: int = 0,
    thermal_mu: float | None = None,
    thermal_sigma: float | None = None,
    metrics_json: str | None = None,
    legacy_grid: bool = False,
) -> pd.DataFrame:
    """Evaluate checkpoint; default is protocol-only (:func:`protocol_corruption` @ sev 1).

    Pass ``legacy_grid=True`` for clean row + arbitrary ``corruptions``/``severities`` sweep.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    ck = _load_checkpoint(ckpt_path, device)
    model, info = _build_model(ck, device)
    threshold = float(threshold_override) if threshold_override is not None else info["threshold"]
    temperature = float(temperature_override) if temperature_override is not None else info["temperature"]

    df_full = _load_index(csv_path)
    if "label" not in df_full.columns and "label_fire" in df_full.columns:
        df_full["label"] = df_full["label_fire"].astype(int)
    df_split = _select_split(df_full, split)
    df_split, dropped = filter_df_existing_paths(df_split, mode=info["mode"])
    if dropped:
        print(f"[robustness] dropped {dropped} rows with missing files for mode={info['mode']!r}")
    if len(df_split) == 0:
        raise SystemExit(f"No rows available for split={split!r} after filtering.")

    proto_c, proto_s = protocol_corruption(str(info["mode"]))
    if legacy_grid:
        if isinstance(corruptions, str):
            corr_names = _resolve_corruptions(corruptions, info["mode"])
        else:
            corr_names = list(corruptions)
        if isinstance(severities, str):
            sev_list = _parse_severities(severities)
        else:
            sev_list = [int(s) for s in severities if 1 <= int(s) <= 3]
    else:
        corr_names = [proto_c]
        sev_list = [int(proto_s)]

    thermal_norm = thermal_norm_from_checkpoint(ck)
    th_mu, th_sigma = resolve_thermal_calibration_or_exit(
        ck=ck,
        thermal_norm=thermal_norm,
        cli_mu=thermal_mu,
        cli_sigma=thermal_sigma,
        metrics_json=metrics_json,
        prog="python -m src.eval.robustness_eval",
    )

    print(
        f"[robustness] ckpt={ckpt_path} mode={info['mode']} family={info['family']} "
        f"backbone={info['backbone']} size={info['size']} threshold={threshold:.3f} T={temperature:.3f} "
        f"thermal_norm={thermal_norm!r}"
    )
    print(f"[robustness] split={split} n={len(df_split)} corruptions={corr_names} severities={sev_list}")

    ds_kw: dict = dict(
        mode=info["mode"], size=info["size"], train=False, thermal_norm=thermal_norm
    )
    if th_mu is not None:
        ds_kw["thermal_mu"] = th_mu
        ds_kw["thermal_sigma"] = float(th_sigma)
    ds = FlameDataset(df_split.reset_index(drop=True), **ds_kw)
    loader = DataLoader(
        ds,
        batch_size=int(bs),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory and device == "cuda"),
    )

    rows: list[dict] = []

    if legacy_grid:
        ys_clean, probs_clean = _eval_loader(model, loader, device, temperature, corruption=None, severity=0)
        rows.append(_row_for(ys_clean, probs_clean, threshold, "clean", 0))
        print(
            f"[robustness][legacy_grid] clean acc={rows[-1]['acc']:.3f} f1={rows[-1]['f1']:.3f} "
            f"recall={rows[-1]['recall']:.3f} fpr={rows[-1]['false_positive_rate']:.3f} "
            f"auc={rows[-1]['auc']:.3f} ap={rows[-1]['ap']:.3f}"
        )

    for name in corr_names:
        fn = CORRUPTIONS[name]
        for sev in sev_list:
            ys, probs = _eval_loader(model, loader, device, temperature, fn, sev)
            row = _row_for(ys, probs, threshold, name, sev)
            rows.append(row)
            print(
                f"[robustness] {name:<22} sev={sev}  acc={row['acc']:.3f} f1={row['f1']:.3f} "
                f"recall={row['recall']:.3f} fpr={row['false_positive_rate']:.3f} "
                f"auc={row['auc']:.3f} ap={row['ap']:.3f}"
            )

    df_out = pd.DataFrame(rows)
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "ckpt": str(ckpt_path),
                    "csv": str(csv_path),
                    "split": split,
                    "model_info": info,
                    "threshold_used": threshold,
                    "temperature_used": temperature,
                    "thermal_norm_ds": thermal_norm,
                    "thermal_mu": th_mu,
                    "thermal_sigma": th_sigma,
                    "corruptions": corr_names,
                    "severities": sev_list,
                    "legacy_grid": bool(legacy_grid),
                    "protocol_default": (not legacy_grid),
                    "n_rows": int(len(df_split)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[robustness] wrote {out_path} (and {meta_path.name})")
    return df_out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Robustness sweep for a trained fire-detection checkpoint."
    )
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--csv", required=True, help="Master index CSV / parquet with split column.")
    ap.add_argument("--split", default="test", choices=["val", "test", "all"], help="Which rows to evaluate.")
    ap.add_argument(
        "--corruptions",
        default="gauss_noise_rgb",
        help=(
            "Legacy-grid only: comma-separated corruption names or 'all'. Ignored unless --legacy-grid."
        ),
    )
    ap.add_argument(
        "--severities",
        default="1",
        help="Legacy-grid only: comma-separated severities (1–3). Ignored unless --legacy-grid.",
    )
    ap.add_argument("--out", default="outputs/robustness_eval.csv", help="Output CSV path.")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=0, choices=[0, 1])
    ap.add_argument("--temperature", type=float, default=None, help="Override calibration T (default: from ckpt).")
    ap.add_argument("--threshold", type=float, default=None, help="Override decision threshold (default: from ckpt).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible noise.")
    ap.add_argument(
        "--metrics_json",
        default=None,
        help="Optional outputs/metrics_*.json path to read thermal_mu/sigma (train_zscore).",
    )
    ap.add_argument(
        "--thermal_mu",
        type=float,
        default=None,
        help="Override thermal mean for train_zscore / global_zscore (use with --thermal_sigma).",
    )
    ap.add_argument(
        "--thermal_sigma",
        type=float,
        default=None,
        help="Override thermal std for train_zscore / global_zscore (use with --thermal_mu).",
    )
    ap.add_argument(
        "--legacy-grid",
        action="store_true",
        help="Include clean baseline row and honor --corruptions/--severities sweep (legacy debugging). "
        "Default: protocol corruption only gauss_noise_rgb@1 or gauss_noise_thermal@1.",
    )
    args = ap.parse_args()

    run_robustness(
        ckpt_path=args.ckpt,
        csv_path=args.csv,
        split=args.split,
        corruptions=args.corruptions,
        severities=args.severities,
        out_csv=args.out,
        bs=int(args.bs),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        temperature_override=args.temperature,
        threshold_override=args.threshold,
        seed=int(args.seed),
        thermal_mu=args.thermal_mu,
        thermal_sigma=args.thermal_sigma,
        metrics_json=args.metrics_json,
        legacy_grid=bool(args.legacy_grid),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
