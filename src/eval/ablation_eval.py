"""Ablation study: RGB-only, thermal-only, modality zeroing, and noisy inputs.

Complements ``src.eval.robustness_eval`` which sweeps corruptions. This module answers
“does fusion actually use both branches?” by comparing forward behaviour on the
same val/test split rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FlameDataset  # noqa: E402
from src.data.path_filter import filter_df_existing_paths  # noqa: E402
from src.models import make_classifier  # noqa: E402
from src.training.metrics import metrics_at_threshold  # noqa: E402


def _load_ckpt(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["label"]).copy()
    if "path_th" not in d.columns and "path_thermal" in d.columns:
        d["path_th"] = d["path_thermal"]
    if "label_fire" not in d.columns:
        d["label_fire"] = d["label"].astype(int)
    return d


def _prob_fire(model: torch.nn.Module, x: torch.Tensor, temperature: float) -> np.ndarray:
    logits = model(x)
    logits = logits / max(1e-6, float(temperature))
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


@torch.inference_mode()
def run_ablation(
    *,
    ckpt_path: str,
    csv_path: str,
    split: str,
    bs: int = 16,
    out_csv: str | Path = "outputs/ablation_eval.csv",
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = _load_ckpt(ckpt_path, device)
    mf = str(ck.get("model_family", "dual_branch_fusion")).lower()
    backbone = str(ck.get("backbone", "resnet50"))
    size = int(ck.get("input_size", 384))
    mode = str(ck.get("mode", "fusion")).lower()
    if mode != "fusion":
        raise SystemExit(f"Ablation fusion suite expects ckpt mode fusion, got {mode!r}")
    Tin = float(ck.get("temperature", 1.0))
    thr = float(ck.get("threshold", 0.5))

    df = pd.read_parquet(csv_path) if csv_path.lower().endswith((".pq", ".parquet")) else pd.read_csv(csv_path)
    df = _prep_df(df)
    if split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)
    df, dropped = filter_df_existing_paths(df, mode="fusion")
    if dropped:
        print(f"[ablation] dropped {dropped} missing-file rows")

    thermal_norm = "percentile"
    ta = ck.get("training_args") if isinstance(ck.get("training_args"), dict) else {}
    tn = ta.get("thermal_norm")
    if isinstance(tn, str) and tn.strip():
        thermal_norm = tn.strip()

    ds = FlameDataset(df, mode="fusion", size=size, train=False, thermal_norm=thermal_norm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    model = make_classifier(mf, backbone, "fusion", num_classes=2, pretrained=False, thermal_init="mean_rgb")
    model.load_state_dict(ck["state"])
    model.to(device).eval()

    ys: list[int] = []
    p_full: list[float] = []
    p_rgb0: list[float] = []
    p_th0: list[float] = []
    p_rgb_noise: list[float] = []
    p_th_noise: list[float] = []

    sigma_n = 0.05
    for xb, yb in dl:
        xb = xb.to(device)
        ys.extend(yb.numpy().tolist())
        p_full.extend(_prob_fire(model, xb, Tin).tolist())

        xx = xb.clone()
        xx[:, :3] = 0
        p_th0.extend(_prob_fire(model, xx, Tin).tolist())

        xx = xb.clone()
        xx[:, 3:] = 0
        p_rgb0.extend(_prob_fire(model, xx, Tin).tolist())

        xx = xb.clone()
        xx[:, :3] = (xx[:, :3] + torch.randn_like(xx[:, :3]) * sigma_n).clamp_(0, 1)
        p_rgb_noise.extend(_prob_fire(model, xx, Tin).tolist())

        xx = xb.clone()
        xx[:, 3:] = (xx[:, 3:] + torch.randn_like(xx[:, 3:]) * sigma_n).clamp_(0, 1)
        p_th_noise.extend(_prob_fire(model, xx, Tin).tolist())

    y_arr = np.asarray(ys, dtype=np.int64)
    rows = []
    suites = [
        ("rgb_full_thermal_full", np.asarray(p_full, np.float32)),
        ("thermal_only_rgb_zero", np.asarray(p_th0, np.float32)),
        ("rgb_only_thermal_zero", np.asarray(p_rgb0, np.float32)),
        ("rgb_gauss_noise", np.asarray(p_rgb_noise, np.float32)),
        ("thermal_gauss_noise", np.asarray(p_th_noise, np.float32)),
    ]
    for name, pr in suites:
        m = metrics_at_threshold(y_arr, pr, thr)
        rows.append({"condition": name, "threshold": thr, "temperature": Tin, **{k: float(v) if not isinstance(v, np.ndarray) else v.tolist() for k, v in m.items()}})
    df_out = pd.DataFrame(rows)
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(outp, index=False)

    meta = outp.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "ckpt": ckpt_path,
                "csv": csv_path,
                "split": split,
                "thermal_norm_ds": thermal_norm,
                "n": int(len(y_arr)),
                "suite": [r[0] for r in suites],
                "note_rgb_only": (
                    "RGB-only scores are inferred by zeroing the thermal channel "
                    "(expected ~random if fusion relies chiefly on RGB)."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[ablation] wrote {outp} (+ {meta.name})")
    return df_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "all"])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="outputs/ablation_eval.csv")
    args = ap.parse_args()
    run_ablation(
        ckpt_path=args.ckpt,
        csv_path=args.csv,
        split=args.split,
        bs=int(args.bs),
        out_csv=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
