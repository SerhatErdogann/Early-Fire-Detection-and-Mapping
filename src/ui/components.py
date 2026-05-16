"""Reusable UI widgets and small layout helpers."""
from __future__ import annotations

from typing import Literal

import streamlit as st


def badge_html(label: str, kind: Literal["ok", "warn", "danger"]) -> str:
    cls = {"ok": "fire-badge-ok", "warn": "fire-badge-warn", "danger": "fire-badge-danger"}[kind]
    return f'<span class="fire-badge {cls}">{label}</span>'


def card_begin() -> None:
    st.markdown('<div class="fire-card">', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_status_header(
    title: str,
    prob_pct: float,
    status_label: str,
    kind: Literal["ok", "warn", "danger"],
) -> None:
    card_begin()
    bhtml = badge_html(status_label, kind)
    st.markdown(
        f'<p class="fire-subtle" style="margin:0">Yangın olasılığı (tahmini)</p>'
        f'<p class="fire-prob-massive" style="color:#eee">%{prob_pct:.0f}</p>'
        f'<p style="margin:0.25rem 0 0 0">{bhtml}</p>'
        f'<p style="margin:0.6rem 0 0 0; font-weight:600; font-size:1.05rem">{title}</p>',
        unsafe_allow_html=True,
    )
    card_end()


def render_metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (k, v) in zip(cols, items):
        with c:
            st.metric(k, v)
