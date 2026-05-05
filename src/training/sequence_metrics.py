"""
Frame-sequence event evaluation for training-time reporting (optional).

Requires per-sequence ordering (``frame_idx`` or similar). If metadata is
missing, callers get ``None`` back (graceful skip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..inference.alarm import ALARM_CONFIRMED, AlarmConfig, AlarmStateMachine

_GROUP_COLS = ("video_id", "video_key", "key")
_FRAME_COLS = ("frame_idx", "frame_index", "frame")


def _pick_group_col(df: pd.DataFrame) -> str | None:
    for c in _GROUP_COLS:
        if c in df.columns:
            return c
    return None


def _pick_frame_col(df: pd.DataFrame) -> str | None:
    for c in _FRAME_COLS:
        if c in df.columns:
            return c
    return None


def compute_sequence_alarm_summary(
    df: pd.DataFrame,
    ys: np.ndarray,
    ps: np.ndarray,
    *,
    prob_threshold_high: float,
    prob_threshold_low: float | None = None,
    confirm_frames: int = 5,
    cooldown_frames: int = 6,
) -> dict | None:
    """
    Per video/key sequence: hysteresis alarm (``AlarmStateMachine``), then summarize
    false-alarm vs missed-fire events vs detection latency vs ground truth labels.

    Returns ``None`` if inputs are malformed. Returns ``{"skipped": True, ...}`` if
    sequence metadata prevents safe evaluation (e.g. multi-row groups without frames).
    """
    if len(df) != len(ys) or len(ys) != len(ps):
        return None
    if len(df) == 0:
        return {"skipped": True, "reason": "empty"}

    gc = _pick_group_col(df)
    if gc is None:
        return {"skipped": True, "reason": "no_group_column"}

    fc = _pick_frame_col(df)
    groups = df[gc].astype(str).to_numpy()
    u = sorted(set(groups.tolist()))

    if fc is None:
        for gid in u:
            if int((groups == gid).sum()) > 1:
                return {"skipped": True, "reason": "missing_frame_column_for_sequences"}

    plow = (
        float(prob_threshold_low)
        if prob_threshold_low is not None
        else max(0.05, float(prob_threshold_high) * 0.55)
    )
    cfg = AlarmConfig(
        high_threshold=float(prob_threshold_high),
        low_threshold=min(plow, float(prob_threshold_high) - 1e-6),
        confirm_frames=int(confirm_frames),
        cooldown_frames=int(cooldown_frames),
    )

    false_alarm_sequences = 0
    missed_fire_sequences = 0
    latencies: list[int] = []

    for gid in u:
        m = groups == gid
        positions = np.flatnonzero(m)
        if fc is None:
            fr_ordered = np.arange(len(positions), dtype=np.float64)
        else:
            fr_raw = pd.to_numeric(df.iloc[positions][fc], errors="coerce").to_numpy(dtype=np.float64)
            order = np.argsort(fr_raw)
            positions = positions[order]
            fr_ordered = fr_raw[order]

        ix = positions
        y_seq = ys[ix].astype(np.int64)
        p_seq = ps[ix].astype(np.float64)

        gt_fire_seq = bool((y_seq == 1).any())

        sm = AlarmStateMachine(cfg)
        confirmed_flags: list[int] = []
        for pj in p_seq:
            st, _ev, _conf, _ = sm.update(decision_prob=float(pj))
            confirmed_flags.append(1 if st == ALARM_CONFIRMED else 0)

        ever_confirmed = any(cf == 1 for cf in confirmed_flags)
        confirmed_idx = None
        for i, cf in enumerate(confirmed_flags):
            if int(cf) == 1:
                confirmed_idx = float(fr_ordered[i])
                break

        if not gt_fire_seq and ever_confirmed:
            false_alarm_sequences += 1
        elif gt_fire_seq and not ever_confirmed:
            missed_fire_sequences += 1
        elif gt_fire_seq and ever_confirmed:
            mask_fire = y_seq == 1
            if mask_fire.any():
                first_fire = float(np.min(fr_ordered[mask_fire]))
                if confirmed_idx is not None:
                    latency = float(confirmed_idx - first_fire)
                    if latency == latency:
                        latencies.append(int(round(latency)))

    return {
        "skipped": False,
        "group_column": gc,
        "frame_column": fc,
        "n_sequences": len(u),
        "false_alarm_event_count": int(false_alarm_sequences),
        "missed_fire_event_count": int(missed_fire_sequences),
        "detection_latency_frames_mean": float(np.mean(latencies)) if latencies else float("nan"),
        "detection_latency_frames_count": len(latencies),
        "alarm_confirm_frames": int(confirm_frames),
        "alarm_prob_high": float(prob_threshold_high),
        "alarm_prob_low": float(plow),
    }
