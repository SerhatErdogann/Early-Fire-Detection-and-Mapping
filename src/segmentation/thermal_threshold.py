# src/segmentation/thermal_threshold.py

import cv2
import numpy as np


def normalize_thermal_frame(thermal_frame):
    """
    Thermal frame'i 0-255 aralığına normalize eder.
    Hem grayscale hem renkli görüntüler için çalışır.
    """

    if thermal_frame is None:
        raise ValueError("thermal_frame is None")

    # Eğer görüntü 3 kanallı gelirse grayscale'e çevir
    if len(thermal_frame.shape) == 3:
        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

    thermal = thermal_frame.astype(np.float32)

    finite_mask = np.isfinite(thermal)
    if not finite_mask.any():
        return np.zeros_like(thermal_frame, dtype=np.uint8)

    valid_pixels = thermal[finite_mask]

    p2 = np.percentile(valid_pixels, 2)
    p98 = np.percentile(valid_pixels, 98)

    if p98 - p2 < 1e-6:
        return np.zeros_like(thermal_frame, dtype=np.uint8)

    thermal = np.clip(thermal, p2, p98)
    thermal = (thermal - p2) / (p98 - p2)
    thermal = (thermal * 255).astype(np.uint8)

    return thermal


def create_fire_mask_from_thermal(
    thermal_frame,
    threshold_value=210,
    min_area=150,
    kernel_size=7,
    threshold_mode="percentile",
    percentile_value=97,
    use_strong_closing=False
):
    """
    Thermal frame üzerinden yangın maskesi çıkarır.

    threshold_mode:
        "fixed"      -> threshold_value kullanır
        "percentile" -> görüntünün en sıcak/parlak üst yüzdesini alır

    percentile_value:
        97 ise görüntünün en parlak %3 kısmı alınır.

    use_strong_closing:
        True olursa yangın parçalarını yatayda güçlü birleştirir.
        False olursa daha sıkı maske üretir.
    """

    thermal_norm = normalize_thermal_frame(thermal_frame)

    if len(thermal_norm.shape) == 3:
        thermal_norm = cv2.cvtColor(thermal_norm, cv2.COLOR_BGR2GRAY)

    if threshold_mode == "percentile":
        threshold_value = np.percentile(thermal_norm, percentile_value)

    _, mask = cv2.threshold(
        thermal_norm,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    mask = mask.astype(np.uint8)

    open_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if use_strong_closing:
        close_kernel = np.ones((9, 35), np.uint8)
    else:
        close_kernel = np.ones((3, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned_mask[labels == label_id] = 255

    return cleaned_mask, thermal_norm