"""Dual-branch RGB + thermal fusion (separate encoders, fused head).

Each modality has its own backbone (pretrained independently), features are
concatenated and fed into a small MLP classifier. This keeps the modality
branches decoupled and tends to reduce RGB-dominance bias that the early
fusion 4-channel variant suffers from.
"""
import torch
import torch.nn as nn

from ..backbones import make_feature_extractor


class DualBranchFusion(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        hidden: int = 512,
        *,
        thermal_init: str = "mean_rgb",
    ):
        super().__init__()
        self.backbone_name = str(backbone or "resnet50").lower()
        self.thermal_init = str(thermal_init or "mean_rgb")
        self.rgb_branch, d_rgb = make_feature_extractor(
            self.backbone_name, 3, pretrained=pretrained, thermal_init="mean_rgb"
        )
        self.th_branch, d_th = make_feature_extractor(
            self.backbone_name, 1, pretrained=pretrained, thermal_init=self.thermal_init
        )
        self.head = nn.Sequential(
            nn.Linear(d_rgb + d_th, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):  # noqa: ARG004
        rgb = x[:, :3]
        th = x[:, 3:4]
        fr = self.rgb_branch(rgb)
        ft = self.th_branch(th)
        fr = fr.view(fr.size(0), -1)
        ft = ft.view(ft.size(0), -1)
        z = torch.cat([fr, ft], dim=1)
        return self.head(z)
