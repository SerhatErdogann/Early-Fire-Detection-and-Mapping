"""Streamlit UI için tek varsayılan video çıkarım profili (ayar seçici yok)."""
from __future__ import annotations

from typing import Any

try:
    from config import INFERENCE_DEFAULT
except Exception:  # pragma: no cover
    INFERENCE_DEFAULT = {}

# Streamlit varsayılan çıkarım argümanları (tek profil; CLI farklı parametre kullanabilir).
DEFAULT_INFER_UI_ARGS: dict[str, Any] = {
    "size": 224,
    "step": 6,
    "smooth_win": 7,
    "ema_alpha": 0.30,
    "tta": True,
    "fp16": True,
    "adaptive_step": True,
    "adaptive_min_step": 2,
    "adaptive_max_step": 12,
    "temporal_guard": True,
    "min_component_area": float(INFERENCE_DEFAULT.get("min_component_area", 0.01) or 0.01),
    "texture_prob_max": float(INFERENCE_DEFAULT.get("texture_prob_max", 0.2) or 0.2),
    "small_fire_boost": float(INFERENCE_DEFAULT.get("small_fire_boost", 1.3) or 1.3),
    "growth_upscale": float(INFERENCE_DEFAULT.get("growth_upscale", 1.2) or 1.2),
    "prob_temporal_blend": 0.2,
    "burst_min_frames": 3,
    "burst_threshold_frac": 1.0,
    "auto_step_long_video": True,
    "stream_buffer_reduce": True,
}


__all__ = ["DEFAULT_INFER_UI_ARGS"]
