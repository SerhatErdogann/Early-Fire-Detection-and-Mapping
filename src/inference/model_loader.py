"""Checkpoint loader shared by inference scripts.

Restores classifiers saved by ``trainer.py``:

- Dual-branch families (plain / gated / attention / mid) via ``make_classifier``
- Early fusion (4-channel single trunk) and RGB/Thermal baselines via ``make_model``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn

try:
    from src.models import FUSION_DUAL_FAMILIES, make_classifier
    from src.models.backbones import make_model
except ImportError:
    from ..models import FUSION_DUAL_FAMILIES, make_classifier
    from ..models.backbones import make_model


def _infer_model_family_from_state(state: dict[str, Any], mf_from_ckpt: str) -> str:
    mf = str(mf_from_ckpt or "").lower().strip()
    if mf:
        return mf
    keys = state.keys()
    if not isinstance(keys, (list, set, tuple)):
        keys = list(keys)

    def _has_prefix(prefix: str) -> bool:
        return any(isinstance(k, str) and k.startswith(prefix) for k in keys)

    if _has_prefix("gate_mlp."):
        return "dual_branch_gated_fusion"
    if _has_prefix("rgb_proj."):
        return "dual_branch_attention_fusion"
    if _has_prefix("mid_fuse.") or _has_prefix("rgb_fx."):
        return "dual_branch_mid_fusion"
    if _has_prefix("rgb_branch."):
        return "dual_branch_fusion"
    return mf


def _thermal_init_from_ckpt(ckpt: dict[str, Any]) -> str:
    ta = ckpt.get("training_args")
    if isinstance(ta, dict) and ta.get("thermal_init") is not None:
        return str(ta.get("thermal_init"))
    return "mean_rgb"


def load_checkpoint(ckpt_path) -> Tuple[nn.Module, str, str, float, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state"]
    raw_mf = ckpt.get("model_family") or ckpt.get("arch") or ""
    model_family = _infer_model_family_from_state(state, str(raw_mf))
    in_ch = int(ckpt.get("in_ch", 3))
    backbone = str(ckpt.get("backbone", "resnet18"))
    tin = _thermal_init_from_ckpt(ckpt)

    if model_family in FUSION_DUAL_FAMILIES:
        model = make_classifier(
            model_family,
            backbone,
            "fusion",
            num_classes=2,
            pretrained=False,
            thermal_init=tin,
        ).to(device)
    else:
        model = make_model(backbone, in_ch, pretrained=False).to(device)
    model.load_state_dict(state)
    model.eval()
    thr = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    mode = ckpt.get("mode", "fusion")
    return model, mode, device, thr, temperature


def route_checkpoint_for_video(
    ckpt_path: str,
    *,
    has_thermal_video: bool,
) -> tuple[str | None, str | None, str | None, str]:
    """Map one UI-selected checkpoint to ``run_video_inference`` arguments.

    RGB-only workflows still use dual-branch checkpoints: ``video.py`` feeds a
    zero thermal plane into the fusion model (see ``mode_used == "rgb"`` branch).

    Returns ``(ckpt_fusion, ckpt_rgb, ckpt_thermal, mode)``.
    """
    p = Path(str(ckpt_path))
    if not p.is_file():
        if has_thermal_video:
            return str(ckpt_path), None, None, "fusion"
        return None, str(ckpt_path), None, "rgb"

    try:
        cpu_ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        cpu_ckpt = torch.load(p, map_location="cpu")

    state = cpu_ckpt.get("state") or {}
    raw_mf = cpu_ckpt.get("model_family") or cpu_ckpt.get("arch") or ""
    mf = _infer_model_family_from_state(state, str(raw_mf))
    mf_l = mf.lower().strip()
    in_ch = int(cpu_ckpt.get("in_ch", 3))

    if mf_l in FUSION_DUAL_FAMILIES:
        return str(p), None, None, ("fusion" if has_thermal_video else "rgb")
    if mf_l == "thermal_baseline" or in_ch == 1:
        return None, None, str(p), "thermal"
    if mf_l == "early_fusion" or in_ch == 4:
        return str(p), None, None, ("fusion" if has_thermal_video else "rgb")
    return None, str(p), None, "rgb"
