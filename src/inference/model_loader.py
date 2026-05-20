"""Checkpoint loader shared by inference scripts.

Restores the production ``dual_branch_gated_fusion`` classifier saved by ``trainer.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn

try:
    from src.models import make_classifier
except ImportError:
    from ..models import make_classifier


class UnsupportedCheckpointError(RuntimeError):
    """Raised when a checkpoint is not a gated dual-branch fusion model."""


def _has_state_prefix(state: dict[str, Any], prefix: str) -> bool:
    return any(isinstance(k, str) and k.startswith(prefix) for k in state.keys())


def _infer_model_family_from_state(state: dict[str, Any], mf_from_ckpt: str) -> str:
    """Return canonical family name or raise if the weights are unsupported."""
    mf = str(mf_from_ckpt or "").lower().strip()
    if mf == "dual_branch_gated_fusion":
        return mf
    if _has_state_prefix(state, "gate_mlp."):
        return "dual_branch_gated_fusion"
    legacy_hits: list[str] = []
    if _has_state_prefix(state, "rgb_proj."):
        legacy_hits.append("dual_branch_attention_fusion")
    if _has_state_prefix(state, "mid_fuse.") or _has_state_prefix(state, "rgb_fx."):
        legacy_hits.append("dual_branch_mid_fusion")
    if _has_state_prefix(state, "rgb_branch.") and not _has_state_prefix(state, "gate_mlp."):
        legacy_hits.append("dual_branch_fusion / non-gated")
    if mf and mf not in ("", "dual_branch_gated_fusion"):
        legacy_hits.append(f"saved tag {mf_from_ckpt!r}")
    if legacy_hits or mf:
        raise UnsupportedCheckpointError(
            "Unsupported checkpoint architecture. This build loads only dual_branch_gated_fusion weights. "
            f"Detected: {', '.join(legacy_hits) or 'unknown'}. Retrain or export with the gated model."
        )
    raise UnsupportedCheckpointError(
        "Checkpoint is missing gated fusion weights (expected keys starting with gate_mlp.) "
        "and model_family was not dual_branch_gated_fusion."
    )


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
    backbone = str(ckpt.get("backbone", "resnet18"))
    tin = _thermal_init_from_ckpt(ckpt)

    model = make_classifier(
        model_family,
        backbone,
        "fusion",
        num_classes=2,
        pretrained=False,
        thermal_init=tin,
    ).to(device)
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

    RGB-only workflows use the gated fusion checkpoint with a zero thermal plane
    (see ``video.py``, ``mode_used == "rgb"``).

    Returns ``(ckpt_fusion, ckpt_rgb, ckpt_thermal, mode)``.
    """
    p = Path(str(ckpt_path))
    if not p.is_file():
        return str(ckpt_path), None, None, ("fusion" if has_thermal_video else "rgb")

    try:
        cpu_ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        cpu_ckpt = torch.load(p, map_location="cpu")

    state = cpu_ckpt.get("state") or {}
    raw_mf = cpu_ckpt.get("model_family") or cpu_ckpt.get("arch") or ""
    _infer_model_family_from_state(state, str(raw_mf))
    return str(p), None, None, ("fusion" if has_thermal_video else "rgb")
