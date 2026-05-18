# src/model_loader.py
from __future__ import annotations

import torch

try:
    from src.inference.model_loader import load_checkpoint
except ImportError:
    from inference.model_loader import load_checkpoint


def _read_checkpoint_meta(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_dual_branch_model(checkpoint_path="outputs/checkpoints/dual_branch.pt", device=None):
    """
    Backward-compatible loader for live/unified pipelines.

    The actual checkpoint restoration is delegated to src.inference.model_loader,
    which supports all current fusion families instead of only the gated variant.
    """
    model, mode, resolved_device, threshold, temperature = load_checkpoint(checkpoint_path)
    ckpt = _read_checkpoint_meta(checkpoint_path, resolved_device)

    return model, {
        "device": resolved_device,
        "threshold": float(ckpt.get("threshold_recommended", threshold)),
        "input_size": int(ckpt.get("input_size", ckpt.get("size", 384))),
        "temperature": float(temperature),
        "class_mapping": ckpt.get("class_mapping"),
        "model_family": ckpt.get("model_family") or ckpt.get("arch"),
        "backbone": ckpt.get("backbone", "resnet18"),
        "mode": mode,
    }


def predict_fire_probability(model, input_tensor, temperature=1.0):
    """
    Modelden fire probability dondurur.
    input_tensor shape: [1, C, H, W]
    """
    with torch.no_grad():
        logits = model(input_tensor)
        if temperature and temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].item())
