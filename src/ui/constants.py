"""UI constants: inference presets for the Streamlit dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferPreset:
    key: str
    title: str
    description: str
    args: dict[str, Any]


try:
    from config import INFERENCE_DEFAULT
except Exception:  # pragma: no cover
    INFERENCE_DEFAULT = {}


PRESETS: list[InferPreset] = [
    InferPreset(
        key="fast",
        title="Hızlı analiz",
        description="Uzun videolarda daha seyrek örnekleme; sonuç hızlı gelir, ince ayrıntı kaçabilir.",
        args={
            "size": 224,
            "step": 10,
            "smooth_win": 5,
            "ema_alpha": 0.25,
            "tta": False,
            "fp16": True,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": 0.0,
            "texture_prob_max": 0.0,
            "small_fire_boost": 1.0,
            "growth_upscale": 1.0,
            "prob_temporal_blend": 0.0,
            "burst_min_frames": 3,
            "burst_threshold_frac": 1.0,
            "auto_step_long_video": True,
            "stream_buffer_reduce": True,
        },
    ),
    InferPreset(
        key="balanced",
        title="Dengeli analiz",
        description="Günlük kullanım için önerilen denge: hız ve güvenilirlik.",
        args={
            "size": 224,
            "step": 6,
            "smooth_win": 7,
            "ema_alpha": 0.30,
            "tta": True,
            "fp16": True,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": float(INFERENCE_DEFAULT.get("min_component_area", 0.01) or 0.01),
            "prob_temporal_blend": 0.2,
            "burst_min_frames": 3,
            "burst_threshold_frac": 1.0,
            "auto_step_long_video": True,
            "stream_buffer_reduce": True,
        },
    ),
    InferPreset(
        key="safe",
        title="Detaylı analiz",
        description="Daha büyük giriş ve daha sık örnekleme; yavaş ama daha ayrıntılı.",
        args={
            "size": 384,
            "step": 4,
            "smooth_win": 9,
            "ema_alpha": 0.35,
            "tta": True,
            "fp16": False,
            "adaptive_step": True,
            "temporal_guard": True,
            "min_component_area": float(INFERENCE_DEFAULT.get("min_component_area", 0.01) or 0.01),
            "prob_temporal_blend": 0.25,
            "burst_min_frames": 4,
            "burst_threshold_frac": 1.0,
            "auto_step_long_video": True,
            "stream_buffer_reduce": True,
        },
    ),
]
