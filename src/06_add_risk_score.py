"""
Compute risk_score from classifier + spatial (CAM/mask) + simple temporal features.
Weights: config RISK_SCORE_WEIGHTS (new) with fallback to RISK_WEIGHTS (legacy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import OUTPUTS_DIR, RISK_WEIGHTS, RISK_SCORE_WEIGHTS, FIRE_EVENT_THR, FIRE_EVENT_MIN_RUN
except ImportError:
    OUTPUTS_DIR = Path("outputs")
    RISK_WEIGHTS = {"prob_fire": 0.60, "intensity_top10": 0.25, "area_heat_gt_0_6": 0.15}
    RISK_SCORE_WEIGHTS = {
        "prob_fire_cal": 0.35,
        "peak_intensity": 0.20,
        "largest_component_area": 0.20,
        "temporal_persistence": 0.15,
        "mask_growth_rate": 0.10,
    }
    FIRE_EVENT_THR = 0.85
    FIRE_EVENT_MIN_RUN = 5

DEFAULT_INP = OUTPUTS_DIR / "video_predictions.csv"
DEFAULT_OUT = OUTPUTS_DIR / "video_predictions_scored.csv"


def _load_temperature_from_ckpt(ckpt_path: Path | None) -> float:
    if ckpt_path is None or not ckpt_path.exists():
        return 1.0
    try:
        import torch

        ck = torch.load(ckpt_path, map_location="cpu")
        return float(ck.get("temperature", 1.0))
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser(description="Add risk_score and fire_event columns to video CSV")
    ap.add_argument("--inp", default=str(DEFAULT_INP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--ckpt", default=None, help="Optional .pt to read temperature for prob scaling metadata only")
    ap.add_argument("--persistence_win", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    ckpt = Path(args.ckpt) if args.ckpt else None
    T = _load_temperature_from_ckpt(ckpt)

    # Without logits, use prob_fire as calibrated proxy (temperature stored for reference)
    df["prob_fire_cal"] = df["prob_fire"].astype(float)

    if "peak_intensity" not in df.columns:
        df["peak_intensity"] = df.get("intensity_top10", df["prob_fire"])
    if "largest_component_area" not in df.columns:
        df["largest_component_area"] = df.get("area_heat_gt_0_6", 0.0)
    if "mask_growth_rate" not in df.columns:
        df["mask_growth_rate"] = df.get("growth_rate", 0.0)

    df = df.sort_values("frame_idx").reset_index(drop=True)
    thr_p = 0.5
    ser = (df["prob_fire"].astype(float) >= thr_p).astype(float)
    df["temporal_persistence"] = ser.rolling(int(args.persistence_win), min_periods=1).mean().fillna(0.0)

    w_new = {k: float(v) for k, v in RISK_SCORE_WEIGHTS.items()}
    df["risk_score"] = (
        w_new.get("prob_fire_cal", 0.35) * df["prob_fire_cal"].astype(float)
        + w_new.get("peak_intensity", 0.2) * df["peak_intensity"].astype(float)
        + w_new.get("largest_component_area", 0.2) * df["largest_component_area"].astype(float)
        + w_new.get("temporal_persistence", 0.15) * df["temporal_persistence"].astype(float)
        + w_new.get("mask_growth_rate", 0.1) * np.maximum(df["mask_growth_rate"].astype(float), 0.0)
    )

    df["risk_score_legacy"] = (
        float(RISK_WEIGHTS.get("prob_fire", 0.6)) * df["prob_fire"].astype(float)
        + float(RISK_WEIGHTS.get("intensity_top10", 0.25)) * df.get("intensity_top10", df["prob_fire"]).astype(float)
        + float(RISK_WEIGHTS.get("area_heat_gt_0_6", 0.15)) * df.get("area_heat_gt_0_6", 0.0).astype(float)
    )

    mx = df["risk_score"].max()
    df["risk_score_norm"] = df["risk_score"] / (mx + 1e-9)

    use_infer_fire = (
        "infer_temporal_applied" in df.columns and int(df["infer_temporal_applied"].iloc[0]) == 1
    )
    if use_infer_fire and "fire_event" in df.columns:
        pass
    else:
        run = 0
        runs = []
        events = []
        probs = df["prob_fire"].to_numpy(dtype=float)
        for p in probs:
            if p >= FIRE_EVENT_THR:
                run += 1
            else:
                run = 0
            runs.append(run)
            events.append(1 if run >= FIRE_EVENT_MIN_RUN else 0)

        df["fire_run_len"] = runs
        df["fire_event"] = events

    meta = {"risk_weights_used": w_new, "temperature_from_ckpt": T}
    df.to_csv(args.out, index=False)
    with open(Path(args.out).with_suffix(".risk_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("✅ Written:", args.out)
    print(df[["frame_idx", "prob_fire", "risk_score", "risk_score_norm", "fire_event"]].head(10))


if __name__ == "__main__":
    main()
