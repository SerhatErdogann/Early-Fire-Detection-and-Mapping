"""
Model backbones for fire classification (RGB, thermal, or RGB+thermal fusion).
ResNet18 with ImageNet init; 4-channel fusion uses RGB weights + mean for thermal.
Optional ResNet50 for higher capacity (better accuracy on difficult drone footage).
"""
import torch
import torch.nn as nn
from torchvision import models

# ResNet50 optional (heavier but more accurate)
try:
    _ResNet50Weights = models.ResNet50_Weights.IMAGENET1K_V2
except Exception:
    _ResNet50Weights = None


def adapt_first_conv(
    model: nn.Module,
    in_ch: int,
    *,
    thermal_init: str = "mean_rgb",
):
    """Replace first conv for thermal stems (in_ch == 1).

    ``thermal_init`` (``in_ch == 1`` only) controls how the thermal stem is
    initialised from pretrained RGB weights:

    - ``mean_rgb`` (default): mean over RGB channels per filter.
    - ``red`` / ``green`` / ``blue``: pick a single RGB channel column.
    - ``kaiming``: random Kaiming init (no reuse of ImageNet conv1).
    """
    old = model.conv1
    model.conv1 = nn.Conv2d(
        in_ch,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    thermal_init = str(thermal_init or "mean_rgb").lower()
    if in_ch == 1 and thermal_init == "kaiming":
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        return model
    with torch.no_grad():
        if in_ch == 1:
            w = old.weight
            if thermal_init in ("red", "r", "ch0"):
                model.conv1.weight[:] = w[:, 0:1]
            elif thermal_init in ("green", "g", "ch1"):
                model.conv1.weight[:] = w[:, 1:2]
            elif thermal_init in ("blue", "b", "ch2"):
                model.conv1.weight[:] = w[:, 2:3]
            else:
                model.conv1.weight[:] = w.mean(dim=1, keepdim=True)
    return model


# Backward-compatible alias
_adapt_first_conv = adapt_first_conv


def _resnet_feature_dim(backbone: str) -> int:
    return 2048 if (backbone or "").lower() == "resnet50" else 512


def make_resnet_feature_extractor(
    backbone: str,
    in_ch: int,
    pretrained: bool = True,
    *,
    thermal_init: str = "mean_rgb",
):
    """ResNet trunk with fc replaced by Identity; returns (module, feature_dim)."""
    backbone = (backbone or "resnet18").lower()
    if backbone == "resnet50":
        weights = _ResNet50Weights if (pretrained and in_ch == 3) else None
        m = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
        m = models.resnet18(weights=weights)
    if in_ch != 3:
        kwargs = dict(thermal_init=thermal_init) if in_ch == 1 else {}
        m = adapt_first_conv(m, in_ch, **kwargs)
    m.fc = nn.Identity()
    return m, _resnet_feature_dim(backbone)


def make_efficientnet_b0_feature_extractor(in_ch: int, pretrained: bool = True):
    """EfficientNet-B0 trunk with classifier replaced by global-pool + Identity.
    Returns (module, feature_dim)."""
    try:
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
    except Exception:
        w = None
    m = models.efficientnet_b0(weights=w)
    if in_ch != 3:
        old = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            if in_ch == 1:
                m.features[0][0].weight[:] = old.weight.mean(dim=1, keepdim=True)
    in_f = int(m.classifier[1].in_features)
    # Keep avgpool, drop the final classifier so we expose pooled features.
    m.classifier = nn.Identity()
    return m, in_f


def make_feature_extractor(
    backbone: str,
    in_ch: int,
    pretrained: bool = True,
    *,
    thermal_init: str = "mean_rgb",
):
    """Unified feature-extractor factory used by dual-branch gated fusion.
    Supports resnet18 / resnet50 / efficientnet_b0."""
    b = (backbone or "resnet18").lower()
    if b in ("efficientnet_b0", "efficientnet-b0", "efficientnetb0"):
        return make_efficientnet_b0_feature_extractor(in_ch, pretrained=pretrained)
    return make_resnet_feature_extractor(
        b, in_ch, pretrained=pretrained, thermal_init=thermal_init
    )


def get_model_config(mode: str):
    """Return (in_channels, default_backbone) stored in fusion checkpoints."""
    _ = mode
    return 4, "resnet50"
