from .backbones import make_model, get_model_config
from .cls.dual_branch_fusion import DualBranchFusion


def make_classifier(
    model_family: str,
    backbone: str,
    mode: str,
    num_classes: int = 2,
    pretrained: bool = True,
):
    """
    model_family: rgb_baseline | thermal_baseline | early_fusion | dual_branch_fusion
    mode: rgb | thermal | fusion (controls input channels for dataset)
    """
    mf = (model_family or "early_fusion").lower()
    if mf == "dual_branch_fusion":
        bb = (backbone or "resnet50").lower()
        allowed = {"resnet18", "resnet50", "efficientnet_b0", "efficientnet-b0", "efficientnetb0"}
        if bb not in allowed:
            raise ValueError(
                "dual_branch_fusion supports backbone resnet18, resnet50 or efficientnet_b0."
            )
        return DualBranchFusion(backbone=bb, num_classes=num_classes, pretrained=pretrained)
    in_ch, _ = get_model_config(mode)
    return make_model(backbone, in_ch, num_classes=num_classes, pretrained=pretrained)


__all__ = ["make_model", "get_model_config", "make_classifier", "DualBranchFusion"]
