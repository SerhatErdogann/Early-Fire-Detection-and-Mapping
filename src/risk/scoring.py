"""
Rule-based, explainable risk scoring helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pick_prob_col(df: pd.DataFrame) -> str:
    return "decision_prob" if "decision_prob" in df.columns else "prob_fire"


def _col_or_scalar(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(float(default), index=df.index, dtype="float64")


def _confidence_band(x: float) -> str:
    if x >= 0.8:
        return "high"
    if x >= 0.5:
        return "medium"
    return "low"


def build_risk_table(
    df: pd.DataFrame,
    risk_weights: dict[str, float],
    persistence_win: int = 7,
    persistence_thr: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy().sort_values("frame_idx").reset_index(drop=True)
    prob_col = _pick_prob_col(out)
    out["prob_fire_cal"] = _col_or_scalar(out, prob_col, default=0.0).astype(float)

    if "peak_intensity" not in out.columns:
        if "intensity_top10" in out.columns:
            out["peak_intensity"] = _col_or_scalar(out, "intensity_top10", default=0.0).astype(float)
        else:
            out["peak_intensity"] = _col_or_scalar(out, prob_col, default=0.0).astype(float)
    if "largest_component_area" not in out.columns:
        out["largest_component_area"] = _col_or_scalar(out, "area_heat_gt_0_6", default=0.0).astype(float)
    if "mask_growth_rate" not in out.columns:
        out["mask_growth_rate"] = _col_or_scalar(out, "growth_rate", default=0.0).astype(float)

    out["frame_pos_norm"] = np.linspace(0.0, 1.0, num=max(1, len(out)), dtype=np.float64)
    ser = (out[prob_col].astype(float) >= float(persistence_thr)).astype(float)
    out["temporal_persistence"] = ser.rolling(int(persistence_win), min_periods=1).mean().fillna(0.0)
    out["prob_trend"] = out[prob_col].astype(float).diff().fillna(0.0).rolling(3, min_periods=1).mean()

    w = {k: float(v) for k, v in risk_weights.items()}
    out["risk_score"] = (
        w.get("prob_fire_cal", 0.35) * out["prob_fire_cal"].astype(float)
        + w.get("peak_intensity", 0.2) * out["peak_intensity"].astype(float)
        + w.get("largest_component_area", 0.2) * out["largest_component_area"].astype(float)
        + w.get("temporal_persistence", 0.15) * out["temporal_persistence"].astype(float)
        + w.get("mask_growth_rate", 0.1) * np.maximum(out["mask_growth_rate"].astype(float), 0.0)
    )
    mx = float(out["risk_score"].max()) if len(out) else 0.0
    out["risk_score_norm"] = out["risk_score"] / (mx + 1e-9)
    out["confidence_band"] = out["risk_score_norm"].map(_confidence_band)
    if "alarm_state" not in out.columns:
        fire_event = _col_or_scalar(out, "fire_event", default=0.0).astype(int)
        out["alarm_state"] = np.where(fire_event == 1, "confirmed", "idle")

    def _reason(row) -> str:
        reasons = []
        if float(row["prob_fire_cal"]) >= 0.7:
            reasons.append("yüksek_olasılık")
        if float(row["temporal_persistence"]) >= 0.6:
            reasons.append("persistence_saglandi")
        if float(row["largest_component_area"]) >= 0.02:
            reasons.append("genis_alan")
        if float(row["peak_intensity"]) >= 0.7:
            reasons.append("termal_destek")
        if float(row["prob_trend"]) < -0.05:
            reasons.append("azalan_trend")
        return "|".join(reasons) if reasons else "dusuk_guven"

    out["risk_reason"] = out.apply(_reason, axis=1)
    meta = {
        "probability_column_used": prob_col,
        "persistence_threshold": float(persistence_thr),
        "weights_used": w,
    }
    return out, meta
