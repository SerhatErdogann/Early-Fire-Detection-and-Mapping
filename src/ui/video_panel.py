"""Left column: video / frame preview and thumbnail grid."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.ui.video_helpers import format_timestamp, nearest_row_by_frame, read_frame_rgb, video_info


def render_dual_preview(
    rgb_path: str | None,
    raw_frame_idx: int,
    df_scored: pd.DataFrame,
    prob_col: str,
) -> None:
    """Show original frame preview and metadata (both panels show same decoded frame)."""
    if not rgb_path:
        st.warning("Önizleme için video yolu yok.")
        return
    info = video_info(rgb_path)
    fps = float(info.get("fps") or 0.0)
    fr = read_frame_rgb(rgb_path, raw_frame_idx)
    row = nearest_row_by_frame(df_scored, raw_frame_idx)
    ts = None
    if row is not None and "timestamp_sec" in row.index:
        ts = float(row["timestamp_sec"]) if pd.notna(row.get("timestamp_sec")) else None
    time_str = format_timestamp(ts, fps, raw_frame_idx)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Video akışı — önizleme")
        if fr is not None:
            st.image(fr, use_container_width=True)
        else:
            st.warning("Kare okunamadı.")
    with c2:
        st.caption("Analiz için örneklenen kare")
        if fr is not None:
            st.image(fr, use_container_width=True)
        else:
            st.warning("—")
    st.markdown(
        f"**Kare numarası:** `{raw_frame_idx}` &nbsp;·&nbsp; **Zaman:** `{time_str}`"
    )
    if row is not None and prob_col in row.index and pd.notna(row.get(prob_col)):
        st.markdown(f"**Bu örnekteki yangın olasılığı:** `{float(row[prob_col]):.1%}`")


def render_frame_cards(
    df_scored: pd.DataFrame,
    rgb_path: str,
    prob_col: str,
    *,
    session_key_selected: str,
    max_cards: int = 48,
) -> None:
    """Thumbnail grid; click sets session_state frame_idx; selectbox for explicit pick."""
    n = len(df_scored)
    if n == 0:
        return

    if session_key_selected not in st.session_state:
        st.session_state[session_key_selected] = int(df_scored["frame_idx"].iloc[0])

    step = max(1, int((n + max_cards - 1) // max_cards))
    row_indices = list(range(0, n, step))[:max_cards]

    st.markdown("##### Kare özetleri")
    st.caption("Karta tıklayın veya listeden kare numarası seçin.")

    cols = st.columns(4)
    for i, di in enumerate(row_indices):
        row = df_scored.iloc[di]
        fi = int(row["frame_idx"])
        p = float(row[prob_col]) if prob_col in row.index else 0.0
        im = read_frame_rgb(rgb_path, fi)
        with cols[i % 4]:
            if im is not None:
                st.image(im, use_container_width=True)
            ts = float(row["timestamp_sec"]) if "timestamp_sec" in row.index else 0.0
            if st.button(
                f"%{100.0 * p:.0f} · {ts:.1f}s\n#{fi}",
                key=f"fc_{session_key_selected}_{fi}",
                use_container_width=True,
            ):
                st.session_state[session_key_selected] = fi
                st.rerun()

    sample_ids = [int(df_scored.iloc[i]["frame_idx"]) for i in row_indices]
    cur = int(st.session_state[session_key_selected])
    if cur not in sample_ids:
        sample_ids = sorted(set(sample_ids + [cur]))
    idx_default = min(range(len(sample_ids)), key=lambda i: abs(sample_ids[i] - cur))
    pick = st.selectbox("Kare numarası (özet liste)", options=sample_ids, index=idx_default)
    st.session_state[session_key_selected] = int(pick)
