"""
Compute risk_score from classifier + spatial (CAM/mask) + simple temporal features.
Weights: config RISK_SCORE_WEIGHTS (new) with fallback to RISK_WEIGHTS (legacy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from src.risk.scoring import build_risk_table

DEFAULT_INP = OUTPUTS_DIR / "video_predictions.csv"
DEFAULT_OUT = OUTPUTS_DIR / "video_predictions_scored.csv"


def _series_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(float(default), index=df.index, dtype="float64")


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

    df = df.sort_values("frame_idx").reset_index(drop=True)
    if "threshold_used" in df.columns:
        thr_p = float(pd.to_numeric(df["threshold_used"], errors="coerce").dropna().median())
    else:
        thr_p = 0.5
    w_new = {k: float(v) for k, v in RISK_SCORE_WEIGHTS.items()}
    df, risk_meta = build_risk_table(
        df,
        risk_weights=w_new,
        persistence_win=int(args.persistence_win),
        persistence_thr=float(thr_p),
    )

    intensity_legacy = (
        _series_or_default(df, "intensity_top10", default=0.0)
        if "intensity_top10" in df.columns
        else _series_or_default(df, "prob_fire", default=0.0)
    )
    area_legacy = _series_or_default(df, "area_heat_gt_0_6", default=0.0)
    df["risk_score_legacy"] = (
        float(RISK_WEIGHTS.get("prob_fire", 0.6)) * _series_or_default(df, "prob_fire", default=0.0)
        + float(RISK_WEIGHTS.get("intensity_top10", 0.25)) * intensity_legacy
        + float(RISK_WEIGHTS.get("area_heat_gt_0_6", 0.15)) * area_legacy
    )

    use_infer_fire = (
        "infer_temporal_applied" in df.columns and int(df["infer_temporal_applied"].iloc[0]) == 1
    )
    if use_infer_fire and "fire_event" in df.columns:
        pass
    else:
        run = 0
        runs = []
        events = []
        prob_col = risk_meta.get("probability_column_used", "prob_fire")
        probs = df[prob_col].to_numpy(dtype=float)
        for p in probs:
            if p >= FIRE_EVENT_THR:
                run += 1
            else:
                run = 0
            runs.append(run)
            events.append(1 if run >= FIRE_EVENT_MIN_RUN else 0)

        df["fire_run_len"] = runs
        df["fire_event"] = events

    meta = {"risk_weights_used": w_new, "temperature_from_ckpt": T, **risk_meta}
    df.to_csv(args.out, index=False)
    with open(Path(args.out).with_suffix(".risk_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("✅ Written:", args.out)
    print(df[["frame_idx", "prob_fire", "risk_score", "risk_score_norm", "fire_event"]].head(10))


if __name__ == "__main__":
    main()
