"""RGB + thermal gated dual-branch classifier (production architecture)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..backbones import make_feature_extractor


class DualBranchGatedFusion(nn.Module):
    """Late fusion with learned soft gates on RGB / thermal embeddings."""

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
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_rgb + d_th, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.head = nn.Sequential(
            nn.Linear(d_rgb + d_th, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        rgb = x[:, :3]
        th = x[:, 3:4]
        fr = self.rgb_branch(rgb).flatten(1)
        ft = self.th_branch(th).flatten(1)
        cat = torch.cat([fr, ft], dim=1)
        logits_gate = self.gate_mlp(cat)
        g = torch.softmax(logits_gate, dim=-1)
        cat_gated = torch.cat([fr * g[:, 0:1], ft * g[:, 1:2]], dim=1)
        logits = self.head(cat_gated)
        self.last_gate_rgb = g[:, 0].detach().mean()
        self.last_gate_thermal = g[:, 1].detach().mean()
        if return_aux:
            return logits, {
                "gate_rgb": g[:, 0],
                "gate_thermal": g[:, 1],
                "logits_gate": logits_gate,
            }
        return logits
