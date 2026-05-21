"""
Model backbones for fire classification (RGB, thermal, or RGB+thermal fusion).
ResNet18 with ImageNet init; 4-channel fusion uses RGB weights + mean for thermal.
Optional ResNet50 for higher capacity (better accuracy on difficult drone footage).
"""
import torch
import torch.nn as nn
from torchvision import models

try:
    _ResNet50Weights = models.ResNet50_Weights.IMAGENET1K_V2
except Exception:
    _ResNet50Weights = None


def adapt_first_conv(model: nn.Module, in_ch: int):
    """Replace first conv for in_ch != 3 (1=thermal, 4=fusion)."""
    old = model.conv1
    model.conv1 = nn.Conv2d(
        in_ch,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        if in_ch == 1:
            model.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)
        elif in_ch == 4:
            model.conv1.weight[:, :3] = old.weight
            model.conv1.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
    return model


def _resnet_feature_dim(backbone: str) -> int:
    return 2048 if (backbone or "").lower() == "resnet50" else 512


def make_resnet_feature_extractor(backbone: str, in_ch: int, pretrained: bool = True):
    backbone = (backbone or "resnet18").lower()
    if backbone == "resnet50":
        weights = _ResNet50Weights if (pretrained and in_ch == 3) else None
        m = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
        m = models.resnet18(weights=weights)
    if in_ch != 3:
        m = adapt_first_conv(m, in_ch)
    m.fc = nn.Identity()
    return m, _resnet_feature_dim(backbone)


def make_efficientnet_b0_feature_extractor(in_ch: int, pretrained: bool = True):
    try:
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
    except Exception:
        w = None
    m = models.efficientnet_b0(weights=w)
    if in_ch != 3:
        old = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False,
        )
        with torch.no_grad():
            if in_ch == 1:
                m.features[0][0].weight[:] = old.weight.mean(dim=1, keepdim=True)
            elif in_ch == 4:
                m.features[0][0].weight[:, :3] = old.weight
                m.features[0][0].weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
    in_f = int(m.classifier[1].in_features)
    m.classifier = nn.Identity()
    return m, in_f


def make_feature_extractor(backbone: str, in_ch: int, pretrained: bool = True):
    b = (backbone or "resnet18").lower()
    if b in ("efficientnet_b0", "efficientnet-b0", "efficientnetb0"):
        return make_efficientnet_b0_feature_extractor(in_ch, pretrained=pretrained)
    return make_resnet_feature_extractor(b, in_ch, pretrained=pretrained)


def make_model(backbone: str, in_ch: int, num_classes: int = 2, pretrained: bool = True):
    backbone = (backbone or "resnet18").lower()
    if backbone == "resnet50":
        weights = _ResNet50Weights if (pretrained and in_ch == 3) else None
        m = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
        m = models.resnet18(weights=weights)
    if in_ch != 3:
        m = adapt_first_conv(m, in_ch)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
