from .backbones import get_model_config, make_model
from .cls.dual_branch_gated_fusion import DualBranchGatedFusion


# Kept as a singleton set for readability in trainer / loaders.
FUSION_DUAL_FAMILIES = frozenset({"dual_branch_gated_fusion"})


def make_classifier(
    model_family: str,
    backbone: str,
    mode: str,
    num_classes: int = 2,
    pretrained: bool = True,
    *,
    thermal_init: str = "mean_rgb",
):
    """Build the RGB+thermal gated dual-branch classifier (``mode`` must be fusion)."""
    mf = (model_family or "dual_branch_gated_fusion").lower().strip()
    if mf != "dual_branch_gated_fusion":
        raise ValueError(
            f"Unsupported model_family={model_family!r}; this project ships only dual_branch_gated_fusion."
        )
    m = (mode or "fusion").lower().strip()
    if m != "fusion":
        raise ValueError(f"Unsupported mode={mode!r}; only fusion (4-channel RGB+thermal) is supported.")
    bb = (backbone or "resnet50").lower()
    allowed = {"resnet18", "resnet50", "efficientnet_b0", "efficientnet-b0", "efficientnetb0"}
    if bb not in allowed:
        raise ValueError("dual_branch_gated_fusion requires resnet18 / resnet50 / efficientnet_b0.")
    tin = str(thermal_init or "mean_rgb")
    return DualBranchGatedFusion(
        backbone=bb,
        num_classes=num_classes,
        pretrained=pretrained,
        thermal_init=tin,
    )


__all__ = [
    "make_model",
    "get_model_config",
    "make_classifier",
    "DualBranchGatedFusion",
    "FUSION_DUAL_FAMILIES",
]
