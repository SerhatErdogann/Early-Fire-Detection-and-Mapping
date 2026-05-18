# src/segmentation/thermal_threshold.py

import cv2
import numpy as np
import warnings


def thermal_to_gray(thermal_frame):
    if thermal_frame is None:
        raise ValueError("thermal_frame is None")

    if len(thermal_frame.shape) == 3:
        if thermal_frame.shape[2] >= 3:
            b = thermal_frame[:, :, 0].astype(np.float32)
            g = thermal_frame[:, :, 1].astype(np.float32)
            r = thermal_frame[:, :, 2].astype(np.float32)
            channel_delta = max(float(np.mean(np.abs(b - g))), float(np.mean(np.abs(g - r))))
            if channel_delta > 2.0:
                warnings.warn(
                    "Thermal frame appears to be color-mapped. Thresholding will use grayscale brightness, "
                    "not calibrated temperature.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

    return thermal_frame


def normalize_thermal_frame(thermal_frame):
    """
    Thermal frame'i 0-255 aralığına normalize eder.
    Hem grayscale hem renkli görüntüler için çalışır.
    """

    thermal_frame = thermal_to_gray(thermal_frame)

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
    kernel_size=11,
    threshold_mode="percentile",
    percentile_value=96,
    use_strong_closing=False,
    dilate_iterations=1
):
    """
    Thermal frame üzerinden yangın maskesi çıkarır.

    threshold_mode:
        "fixed"      -> raw thermal/grayscale değerinde threshold_value kullanır
        "absolute"   -> fixed ile aynı; okunabilirlik için alias
        "percentile" -> raw thermal/grayscale görüntünün üst yüzdesini alır
        "hybrid"     -> percentile eşiği ile threshold_value değerinin büyüğünü alır

    percentile_value:
        97 -> görüntünün en parlak üst %3 kısmını alır
        96 -> görüntünün en parlak üst %4 kısmını alır
        95 -> görüntünün en parlak üst %5 kısmını alır

    min_area:
        Küçük gürültü parçalarını silmek için minimum piksel alanı.

    kernel_size:
        Yangın parçalarını birleştirmek için kullanılan closing kernel boyutu.
        Parçalı görünüm varsa artır:
            7  -> hafif
            9  -> orta
            11 -> önerilen
            13 -> güçlü
            15 -> çok güçlü

    use_strong_closing:
        True olursa daha güçlü birleştirme yapar.

    dilate_iterations:
        Parçaları biraz büyütüp yakın bölgeleri birleştirir.
        0 -> genişletme yok
        1 -> önerilen
        2 -> daha güçlü ama alanı fazla büyütebilir
    """

    thermal_gray = thermal_to_gray(thermal_frame)
    thermal_raw = thermal_gray.astype(np.float32)
    finite_mask = np.isfinite(thermal_raw)
    thermal_norm = normalize_thermal_frame(thermal_gray)

    if not finite_mask.any():
        return np.zeros_like(thermal_norm, dtype=np.uint8), thermal_norm

    valid_pixels = thermal_raw[finite_mask]

    # Threshold değerini belirle
    if threshold_mode == "percentile":
        threshold_value = float(np.percentile(valid_pixels, percentile_value))

    elif threshold_mode in ("fixed", "absolute"):
        threshold_value = float(threshold_value)

    elif threshold_mode == "hybrid":
        percentile_threshold = float(np.percentile(valid_pixels, percentile_value))
        threshold_value = max(float(threshold_value), percentile_threshold)

    else:
        raise ValueError(
            "threshold_mode sadece 'fixed', 'absolute', 'percentile' veya 'hybrid' olabilir."
        )

    # Sıcak bölgeleri binary maskeye çevir
    mask = ((thermal_raw >= float(threshold_value)) & finite_mask).astype(np.uint8) * 255

    # 1) Küçük tekil gürültüleri temizle
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    pre_min_area = max(1, int(min_area) // 4)
    mask = remove_small_components(mask, pre_min_area)

    # 2) Parçalı yangın alanlarını birleştir
    if use_strong_closing:
        close_size = max(kernel_size, 15)
    else:
        close_size = kernel_size

    # Kernel boyutu tek sayı olsun
    if close_size % 2 == 0:
        close_size += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # 3) Yakın parçalar hâlâ kopuksa biraz genişlet
    if dilate_iterations > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.dilate(
            mask,
            dilate_kernel,
            iterations=dilate_iterations
        )

    # 4) Connected component ile küçük alanları temizle
    cleaned_mask = remove_small_components(mask, min_area)

    return cleaned_mask, thermal_norm


def remove_small_components(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            cleaned_mask[labels == label_id] = 255

    return cleaned_mask
