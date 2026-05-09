"""Video helpers for the dashboard (preview frames without loading whole video into RAM)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def video_info(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "Video açılamadı."}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"ok": True, "fps": fps, "frame_count": frame_count, "width": w, "height": h}


@st.cache_data(show_spinner=False)
def read_frame_rgb(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Return HxWx3 uint8 RGB or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def nearest_row_by_frame(df_: pd.DataFrame, frame_idx: int) -> pd.Series | None:
    if df_.empty or "frame_idx" not in df_.columns:
        return None
    ff = pd.to_numeric(df_["frame_idx"], errors="coerce")
    if ff.isna().all():
        return None
    idx_near = (ff - float(frame_idx)).abs().idxmin()
    return df_.loc[idx_near]


def format_timestamp(sec: float | None, fps: float, frame_idx: int) -> str:
    if sec is not None and sec == sec and sec >= 0:
        m = int(sec // 60)
        s = sec - m * 60
        return f"{m:d}:{s:05.2f}"
    if fps and fps > 1e-6:
        return f"{float(frame_idx) / float(fps):.2f} s"
    return "—"
