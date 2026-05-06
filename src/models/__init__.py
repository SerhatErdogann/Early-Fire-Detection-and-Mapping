from .backbones import bare_resnet_for_features, make_model, get_model_config
from .cls.dual_branch_fusion import DualBranchFusion
from .cls.fusion_variants import (
    DualBranchAttentionFusion,
    DualBranchGatedFusion,
    DualBranchMidFusion,
)


FUSION_DUAL_FAMILIES = frozenset(
    {
        "dual_branch_fusion",
        "dual_branch_gated_fusion",
        "dual_branch_attention_fusion",
        "dual_branch_mid_fusion",
    }
)


def make_classifier(
    model_family: str,
    backbone: str,
    mode: str,
    num_classes: int = 2,
    pretrained: bool = True,
    *,
    thermal_init: str = "mean_rgb",
):
    """
    model_family: rgb_baseline | thermal_baseline | early_fusion |
        dual_branch_fusion | dual_branch_gated_fusion |
        dual_branch_attention_fusion | dual_branch_mid_fusion
    mode: rgb | thermal | fusion (controls input channels for dataset)
    """
    mf = (model_family or "early_fusion").lower()
    bb = (backbone or "resnet50").lower()
    allowed = {"resnet18", "resnet50", "efficientnet_b0", "efficientnet-b0", "efficientnetb0"}
    tin = str(thermal_init or "mean_rgb")

    if mf == "dual_branch_fusion":
        if bb not in allowed:
            raise ValueError(
                "dual_branch_fusion supports backbone resnet18, resnet50 or efficientnet_b0."
            )
        return DualBranchFusion(
            backbone=bb,
            num_classes=num_classes,
            pretrained=pretrained,
            thermal_init=tin,
        )
    if mf == "dual_branch_gated_fusion":
        if bb not in allowed:
            raise ValueError("dual_branch_gated_fusion requires resnet18 / resnet50 / efficientnet_b0.")
        return DualBranchGatedFusion(
            backbone=bb,
            num_classes=num_classes,
            pretrained=pretrained,
            thermal_init=tin,
        )
    if mf == "dual_branch_attention_fusion":
        if bb not in allowed:
            raise ValueError("dual_branch_attention_fusion requires resnet18 / resnet50 / efficientnet_b0.")
        return DualBranchAttentionFusion(
            backbone=bb,
            num_classes=num_classes,
            pretrained=pretrained,
            thermal_init=tin,
        )
    if mf == "dual_branch_mid_fusion":
        if bb not in ("resnet18", "resnet50"):
            raise ValueError("dual_branch_mid_fusion only supports resnet18 or resnet50.")
        return DualBranchMidFusion(
            backbone=bb,
            num_classes=num_classes,
            pretrained=pretrained,
            thermal_init=tin,
        )

    in_ch, _ = get_model_config(mode)
    return make_model(backbone, in_ch, num_classes=num_classes, pretrained=pretrained)


__all__ = [
    "make_model",
    "get_model_config",
    "make_classifier",
    "DualBranchFusion",
    "FUSION_DUAL_FAMILIES",
]
