# src/frame_preprocess.py

import cv2
import numpy as np
import torch


def normalize_rgb(rgb_frame):
    """
    RGB/BGR frame'i 0-1 aralığına getirir.
    OpenCV BGR okuduğu için RGB'ye çevirir.
    """

    if rgb_frame is None:
        raise ValueError("rgb_frame is None")

    if len(rgb_frame.shape) == 2:
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)

    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = rgb_frame.astype(np.float32) / 255.0

    return rgb_frame


def normalize_thermal_for_model(thermal_frame):
    """
    Thermal frame'i model için 0-1 aralığına normalize eder.
    """

    if thermal_frame is None:
        raise ValueError("thermal_frame is None")

    if len(thermal_frame.shape) == 3:
        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)

    thermal = thermal_frame.astype(np.float32)

    finite_mask = np.isfinite(thermal)

    if not finite_mask.any():
        return np.zeros_like(thermal, dtype=np.float32)

    valid_pixels = thermal[finite_mask]

    p2 = np.percentile(valid_pixels, 2)
    p98 = np.percentile(valid_pixels, 98)

    if p98 - p2 < 1e-6:
        return np.zeros_like(thermal, dtype=np.float32)

    thermal = np.clip(thermal, p2, p98)
    thermal = (thermal - p2) / (p98 - p2)

    return thermal.astype(np.float32)


def build_fusion_tensor(
    rgb_frame,
    thermal_frame,
    input_size=384,
    device="cpu"
):
    """
    RGB + thermal frame'i modele uygun 4 kanallı tensor'a çevirir.

    Output shape:
        [1, 4, input_size, input_size]
    """

    rgb_resized = cv2.resize(
        rgb_frame,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR
    )

    thermal_resized = cv2.resize(
        thermal_frame,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR
    )

    rgb_norm = normalize_rgb(rgb_resized)
    thermal_norm = normalize_thermal_for_model(thermal_resized)

    thermal_norm = np.expand_dims(thermal_norm, axis=-1)

    fusion = np.concatenate(
        [rgb_norm, thermal_norm],
        axis=-1
    )

    fusion = np.transpose(fusion, (2, 0, 1))

    tensor = torch.from_numpy(fusion).float().unsqueeze(0)
    tensor = tensor.to(device)

    return tensor