"""Kullanıcı arayüzünde gösterilecek olasılık biçimi (grafikleri abartılı %100 yapmadan)."""
from __future__ import annotations

# Ham veri iş mantığıyla paylaşılan üst eşik; yalnızca gösterim için tavan.
DISPLAY_PROB_CAP = 0.97


def cap_probability_for_chart(p_raw: float) -> float:
    return float(min(max(float(p_raw), 0.0), DISPLAY_PROB_CAP))


def probability_meter_percent(p_raw: float) -> float:
    """Büyük rakam için 0–100 ölçeği (yüksekler tavanda)."""
    return cap_probability_for_chart(p_raw) * 100.0


def probability_label_percent(p_raw: float) -> str:
    """Metin etiketi: çok yüksekler '>97%' olarak."""
    if float(p_raw) > DISPLAY_PROB_CAP:
        return ">97%"
    return f"{cap_probability_for_chart(p_raw) * 100.0:.0f}%"


__all__ = [
    "DISPLAY_PROB_CAP",
    "cap_probability_for_chart",
    "probability_meter_percent",
    "probability_label_percent",
]
