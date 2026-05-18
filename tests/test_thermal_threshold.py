import numpy as np

from src.segmentation.temporal_smoothing import TemporalMaskSmoother
from src.segmentation.thermal_threshold import create_fire_mask_from_thermal


def test_percentile_threshold_keeps_max_pixels_when_threshold_equals_max():
    frame = np.zeros((20, 20), dtype=np.uint8)
    frame[5:15, 5:15] = 255

    mask, _ = create_fire_mask_from_thermal(
        frame,
        threshold_mode="percentile",
        percentile_value=100,
        min_area=20,
        kernel_size=3,
        dilate_iterations=0,
    )

    assert int((mask > 0).sum()) >= 20


def test_hybrid_threshold_uses_absolute_floor_for_low_contrast_frames():
    frame = np.zeros((20, 20), dtype=np.uint8)
    frame[5:15, 5:15] = 180

    mask, _ = create_fire_mask_from_thermal(
        frame,
        threshold_mode="hybrid",
        threshold_value=210,
        percentile_value=95,
        min_area=20,
        kernel_size=3,
        dilate_iterations=0,
    )

    assert int((mask > 0).sum()) == 0


def test_temporal_smoother_waits_for_min_history():
    smoother = TemporalMaskSmoother(history_size=3, vote_threshold=0.5, min_history=2)
    mask = np.ones((5, 5), dtype=np.uint8) * 255

    first = smoother.update(mask)
    second = smoother.update(mask)

    assert int(first.sum()) == 0
    assert int(second.sum()) > 0
