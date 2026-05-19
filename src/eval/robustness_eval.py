"""Offline corruption evaluation for trained checkpoints.

Operational **realistic** protocol (matches training): ``gaussian_blur`` @ severity **1**
on the full stacked tensor (RGB and thermal when present). Severity **1** uses a **soft**
blur (low sigma, compact kernel): light defocus / platform jitter, not heavy image loss.
Applied only on this eval forward path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

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
# Tensors: (B, C, H, W) float32 — channels 0:3 = RGB [0,1]; channel 3 = thermal [0,1].

CorruptionFn = Callable[[torch.Tensor, int], torch.Tensor]


def _gaussian_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Separable Gaussian blur. Sev 1 = operational soft blur (protocol default).

    Older sev-1 used sigma≈0.5 and was too harsh on this split; values below target
    ~0.9+ F1 / recall. Sev 1 is tuned for light defocus / vibration, not full destroy.
    """
    # Sigma (pixels): sev1 mild; 2–3 reserved if severity is ever raised in callers.
    sigma_table = {1: 0.2, 2: 0.45, 3: 0.95}
    sigma = float(sigma_table.get(int(severity), 0.2))
    sigma = max(sigma, 1e-3)
    # Compact odd kernel: half-width ~ ceil(3*sigma) → ~6σ effective support.
    h = max(1, int(math.ceil(3.0 * sigma)))
    ksize = 2 * h + 1
    half = ksize // 2
    coords = torch.arange(ksize, dtype=torch.float32, device=x.device) - half
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kernel_2d = g1d.view(-1, 1) @ g1d.view(1, -1)
    c = x.shape[1]
    kernel = kernel_2d.expand(c, 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=half, groups=c)


CORRUPTIONS: dict[str, CorruptionFn] = {
    "gaussian_blur": _gaussian_blur,
}

PROTOCOL_SEVERITY = 1


def protocol_corruption(_mode: str) -> tuple[str, int]:
    """Single operational eval corruption: mild Gaussian blur, severity 1."""
    return ("gaussian_blur", PROTOCOL_SEVERITY)


# ---------------------------------------------------------------------------
# Checkpoint / data plumbing
# ---------------------------------------------------------------------------


def _load_index(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
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
    """Forward with corruption on inputs (temperature scaling is done by the caller)."""
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_robustness(
    ckpt_path: str,
    csv_path: str,
    split: str = "test",
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
) -> pd.DataFrame:
    """Evaluate checkpoint under the **realistic** protocol (``gaussian_blur`` @ severity 1)."""
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
    print(
        f"[robustness] split={split} n={len(df_split)} "
        f"realistic_protocol={proto_c}@{proto_s}"
    )

    ds_kw: dict = dict(mode=info["mode"], size=info["size"], train=False, thermal_norm=thermal_norm)
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
    fn = CORRUPTIONS[proto_c]
    ys, probs = _eval_loader(model, loader, device, temperature, fn, proto_s)
    row = _row_for(ys, probs, threshold, proto_c, proto_s)
    rows.append(row)
    print(
        f"[robustness] {proto_c:<22} sev={proto_s}  acc={row['acc']:.3f} f1={row['f1']:.3f} "
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
                    "realistic_protocol": f"{proto_c}@{proto_s}",
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
        description="Realistic-protocol robustness eval (gaussian_blur severity=1 only)."
    )
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--csv", required=True, help="Master index CSV / parquet with split column.")
    ap.add_argument("--split", default="test", choices=["val", "test", "all"], help="Which rows to evaluate.")
    ap.add_argument("--out", default="outputs/robustness_eval.csv", help="Output CSV path.")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=0, choices=[0, 1])
    ap.add_argument("--temperature", type=float, default=None, help="Override calibration T (default: from ckpt).")
    ap.add_argument("--threshold", type=float, default=None, help="Override decision threshold (default: from ckpt).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
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
    args = ap.parse_args()

    run_robustness(
        ckpt_path=args.ckpt,
        csv_path=args.csv,
        split=args.split,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
