import cv2
import numpy as np


def prep_rgb(frame_bgr, size=384):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return arr, rgb


def prep_thermal(frame_bgr_or_gray, size=384):
    if frame_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr_or_gray
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    norm = (gray - lo) / (hi - lo + 1e-6)
    norm = np.clip(norm, 0, 1)
    return norm[None, ...], gray
