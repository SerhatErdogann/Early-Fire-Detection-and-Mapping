import cv2
import numpy as np


def prep_rgb(frame_bgr, size=384):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return arr, rgb


def prep_thermal(frame_bgr_or_gray, size=384):
    """
    Normalize a video thermal frame to the same distribution the training
    Dataset produces. Keep this in sync with
    `src/data/dataset.py::thermal_to_norm01` (percentile path, p2–p98, NaN/inf
    guard). Input may be BGR (3-channel) or single-channel; output is
    (1, size, size) in [0, 1].
    """
    if frame_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr_or_gray
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    finite_mask = np.isfinite(gray)
    if finite_mask.any():
        fill = float(np.median(gray[finite_mask]))
    else:
        fill = 0.0
    gray = np.where(finite_mask, gray, fill).astype(np.float32)
    lo = float(np.percentile(gray, 2.0))
    hi = float(np.percentile(gray, 98.0))
    if hi - lo < 1e-6:
        norm = np.zeros_like(gray, dtype=np.float32)
    else:
        norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm[None, ...], gray
