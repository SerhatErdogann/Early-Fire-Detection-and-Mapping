"""Right-hand panel: Turkish status text and explanations."""
from __future__ import annotations

import streamlit as st

from src.ui.components import render_status_header
from src.ui.display_format import probability_meter_percent


def classify_ui_status(
    prob: float,
    alarm_esigi: float,
    inceleme_esigi: float,
    alarm_state: str | None,
) -> tuple[str, str, str]:
    """
    Returns (kullanıcı_başlığı, kısa_etiket, kind) for styling; kind in ok|warn|danger.
    """
    st_al = (alarm_state or "").lower()
    if st_al == "confirmed" or prob >= float(alarm_esigi):
        return "Yangın riski yüksek", "Üst düzey uyarı", "danger"
    if prob >= float(inceleme_esigi) or st_al in ("suspected",):
        return "İnceleme gerekli", "Dikkat — kontrol edin", "warn"
    return "Yangın yok", "Güvenli bölge", "ok"


def turkish_caption_for_row(
    prob: float,
    alarm_esigi: float,
    inceleme_esigi: float,
    prob_slope: float | None,
    alarm_state: str | None,
) -> str:
    """Short user-facing explanation in Turkish."""
    st_al = (alarm_state or "").lower()
    if st_al == "confirmed" or prob >= float(alarm_esigi):
        return (
            "Model bu karede yangına güçlü biçimde benzeyen parlama/ısı örüntüsü gördü. "
            "Kesin teşhis için görüntüyü uzman gözüyle doğrulayın."
        )
    if prob >= float(inceleme_esigi) or st_al == "suspected":
        return (
            "Orta düzey bir uyarı. Güneş yansıması veya parlamalar yanlış alarm üretebilir; "
            "çevreyi yakından kontrol etmek gerekir."
        )
    if prob_slope is not None and prob_slope > 0.02:
        return (
            "Son karelerde olasılık yükseliyor; dikkatli incelenmeli. "
            "Henüz kesin yangın demek için yeterli değil."
        )
    return "Model bu karede belirgin bir yangın paterni raporlamadı."


def render_live_panel(
    prob_raw: float,
    alarm_esigi: float,
    inceleme_esigi: float,
    alarm_state: str | None,
    caption: str,
) -> None:
    title, badge, kind = classify_ui_status(prob_raw, alarm_esigi, inceleme_esigi, alarm_state)
    render_status_header(
        title,
        probability_meter_percent(prob_raw),
        badge,
        kind,
    )
    st.info(caption)
