"""İzleme: tek canlı görüntü (operasyon görünümü)."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.ui.video_helpers import format_timestamp, nearest_row_by_frame, read_frame_rgb, video_info


def render_live_frame_preview(
    rgb_path: str | None,
    raw_frame_idx: int,
    df_scored: pd.DataFrame | None,
) -> None:
    """Operasyon görünümü: tek görüntü + kısa zaman bilgisi (kare no teknik olarak expander’a)."""
    if not rgb_path:
        st.warning("Önizleme için video yolu yok.")
        return
    info = video_info(rgb_path)
    fps = float(info.get("fps") or 0.0)
    fr = read_frame_rgb(rgb_path, raw_frame_idx)
    ts = None
    row = nearest_row_by_frame(df_scored, raw_frame_idx) if df_scored is not None and not df_scored.empty else None
    if row is not None and "timestamp_sec" in row.index:
        ts = float(row["timestamp_sec"]) if pd.notna(row.get("timestamp_sec")) else None
    time_str = format_timestamp(ts, fps, raw_frame_idx)
    if fr is not None:
        st.image(fr, use_container_width=True)
        st.markdown(f"<p style='margin-top:0.35rem;color:#b8c5d9;font-weight:600'>{time_str}</p>", unsafe_allow_html=True)
    else:
        st.warning("Bu konum için kare okunamadı.")
