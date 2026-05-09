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


def render_preview_with_frame_arrows(
    rgb_path: str | None,
    df_scored: pd.DataFrame | None,
    *,
    session_key_selected: str,
) -> None:
    """Çıktı kare sırasına göre ◀ ▶ ile gezinme (analiz tablosundaki ``frame_idx`` sırası)."""
    if not rgb_path:
        st.warning("Önizleme için video yolu yok.")
        return
    if df_scored is None or df_scored.empty or "frame_idx" not in df_scored.columns:
        render_live_frame_preview(rgb_path, 0, df_scored)
        return

    idx_series = pd.to_numeric(df_scored["frame_idx"], errors="coerce").dropna().astype(int)
    seq = sorted(idx_series.unique().tolist())
    if not seq:
        render_live_frame_preview(rgb_path, 0, df_scored)
        return

    safe = "".join(ch if ch.isalnum() else "_" for ch in session_key_selected)
    nav_prev = f"nav_prev_{safe}"
    nav_next = f"nav_next_{safe}"

    if session_key_selected not in st.session_state:
        st.session_state[session_key_selected] = int(seq[0])

    cur = int(st.session_state[session_key_selected])
    if cur not in seq:
        snapped = seq[min(range(len(seq)), key=lambda i: abs(seq[i] - cur))]
        st.session_state[session_key_selected] = snapped
        cur = snapped
    pos = seq.index(cur)

    c_left, c_mid, c_right = st.columns([0.1, 0.72, 0.1])

    with c_left:
        go_prev = st.button(
            "◀",
            key=nav_prev,
            help="Önceki analiz karesi",
            disabled=pos <= 0,
            use_container_width=True,
        )

    with c_right:
        go_next = st.button(
            "▶",
            key=nav_next,
            help="Sonraki analiz karesi",
            disabled=pos >= len(seq) - 1,
            use_container_width=True,
        )

    if go_prev and pos > 0:
        st.session_state[session_key_selected] = int(seq[pos - 1])
    if go_next and pos < len(seq) - 1:
        st.session_state[session_key_selected] = int(seq[pos + 1])

    cur = int(st.session_state[session_key_selected])
    if cur not in seq:
        snapped = seq[min(range(len(seq)), key=lambda i: abs(seq[i] - cur))]
        st.session_state[session_key_selected] = snapped
        cur = snapped
    pos = seq.index(cur)

    with c_mid:
        render_live_frame_preview(rgb_path, cur, df_scored)

    st.caption(
        f"Analiz sırasında **{pos + 1} / {len(seq)}** kayıt — **kare {cur}** (◀ ▶ çıktı sırasına göre ilerler)"
    )


def render_frame_cards(
    df_scored: pd.DataFrame,
    rgb_path: str,
    prob_col: str,
    *,
    session_key_selected: str,
    max_cards: int = 48,
) -> None:
    """Teknik görünüm: küçük kare özetleri — tıklanınca önizleme çerçevesi seçilir."""
    n = len(df_scored)
    if n == 0:
        return

    if session_key_selected not in st.session_state:
        st.session_state[session_key_selected] = int(df_scored["frame_idx"].iloc[0])

    step = max(1, int((n + max_cards - 1) // max_cards))
    row_indices = list(range(0, n, step))[:max_cards]

    st.caption("Kare özetleri (teknik) — karta tıklayın veya listeden seçin.")

    cols = st.columns(4)
    for i, di in enumerate(row_indices):
        row = df_scored.iloc[di]
        fi = int(row["frame_idx"])
        p = float(row[prob_col]) if prob_col in row.index else 0.0
        im = read_frame_rgb(rgb_path, fi) if rgb_path else None
        with cols[i % 4]:
            if im is not None:
                st.image(im, use_container_width=True)
            ts = float(row["timestamp_sec"]) if "timestamp_sec" in row.index else 0.0
            label = f"%{100.0 * min(p, 0.97):.0f} · {ts:.1f}s\n#{fi}"
            if p > 0.97:
                label = f">97% · {ts:.1f}s\n#{fi}"
            if st.button(label, key=f"fc_{session_key_selected}_{fi}", use_container_width=True):
                st.session_state[session_key_selected] = fi
                st.rerun()

    sample_ids = [int(df_scored.iloc[i]["frame_idx"]) for i in row_indices]
    cur = int(st.session_state[session_key_selected])
    if cur not in sample_ids:
        sample_ids = sorted(set(sample_ids + [cur]))
    idx_default = min(range(len(sample_ids)), key=lambda j: abs(sample_ids[j] - cur))
    pick = st.selectbox("Özet listeden kare", options=sample_ids, index=idx_default)
    st.session_state[session_key_selected] = int(pick)
