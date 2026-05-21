# src/frame_preprocess.py
from __future__ import annotations

import numpy as np
import torch

try:
    from src.inference.preprocess import prep_rgb, prep_thermal
except ImportError:
    from inference.preprocess import prep_rgb, prep_thermal


def normalize_rgb(rgb_frame):
    """Normalize RGB/BGR frame using the shared inference preprocessing."""
    arr, _ = prep_rgb(rgb_frame, size=rgb_frame.shape[0])
    return np.transpose(arr, (1, 2, 0))


def normalize_thermal_for_model(thermal_frame):
    """Normalize thermal frame using the shared inference preprocessing."""
    arr, _ = prep_thermal(thermal_frame, size=thermal_frame.shape[0])
    return arr[0]


def build_fusion_tensor(rgb_frame, thermal_frame, input_size=384, device="cpu"):
    """
    Convert RGB + thermal frames into the fusion model tensor.

    Output shape: [1, 4, input_size, input_size]
    """
    rgb_arr, _ = prep_rgb(rgb_frame, size=input_size)
    thermal_arr, _ = prep_thermal(thermal_frame, size=input_size)
    fusion = np.concatenate([rgb_arr, thermal_arr], axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(fusion)).float().unsqueeze(0)
    return tensor.to(device)
