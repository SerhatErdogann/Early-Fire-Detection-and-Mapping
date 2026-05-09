"""Alternative RGB–thermal fusion heads (gated / attention / mid-level fusion)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.feature_extraction import create_feature_extractor
except Exception:  # pragma: no cover
    create_feature_extractor = None

from ..backbones import bare_resnet_for_features, make_feature_extractor


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


class DualBranchAttentionFusion(nn.Module):
    """Projects RGB / thermal embeddings to a shared space then self-attention pool."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        hidden: int = 256,
        *,
        thermal_init: str = "mean_rgb",
        attn_heads: int = 4,
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
        h = int(hidden)
        heads = max(1, min(int(attn_heads), h))
        if h % heads != 0:
            h = h - (h % heads) + heads  # bump to divisible
            h = max(h, heads * 8)
        self.embed_dim = h
        self.rgb_proj = nn.Linear(d_rgb, h)
        self.th_proj = nn.Linear(d_th, h)
        self.mha = nn.MultiheadAttention(
            embed_dim=h, num_heads=heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(h, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        rgb = x[:, :3]
        th = x[:, 3:4]
        fr = self.rgb_branch(rgb).flatten(1)
        ft = self.th_branch(th).flatten(1)
        t0 = torch.tanh(self.rgb_proj(fr)).unsqueeze(1)
        t1 = torch.tanh(self.th_proj(ft)).unsqueeze(1)
        tok = torch.cat([t0, t1], dim=1)
        attn_out, _ = self.mha(tok, tok, tok)
        fused = self.norm(attn_out.mean(dim=1))
        logits = self.head(fused)
        if return_aux:
            return logits, {"attn_tokens_shape": tuple(attn_out.shape)}
        return logits


class DualBranchMidFusion(nn.Module):
    """Fuse spatial features after ResNet ``layer3`` + deep embeddings."""

    _L3_CHANNELS = {"resnet18": 256, "resnet50": 1024}

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
        if create_feature_extractor is None:
            raise ImportError(
                "torchvision.models.feature_extraction is required for mid fusion "
                "(upgrade torchvision)."
            )
        bb = str(backbone or "resnet50").lower()
        if bb == "efficientnet_b0" or "efficient" in bb:
            raise ValueError("DualBranchMidFusion currently supports only resnet18 / resnet50.")
        key = bb if bb in DualBranchMidFusion._L3_CHANNELS else "resnet50"
        l3_ch = DualBranchMidFusion._L3_CHANNELS[key]
        d_deep = _resnet_dim(key)

        self.backbone_name = bb
        self.thermal_init = str(thermal_init or "mean_rgb")
        rgb_body = bare_resnet_for_features(bb, 3, pretrained=pretrained, thermal_init="mean_rgb")
        th_body = bare_resnet_for_features(
            bb, 1, pretrained=pretrained, thermal_init=self.thermal_init
        )
        nodes = {"layer3": "l3", "flatten": "deep"}
        self.rgb_fx = create_feature_extractor(rgb_body, return_nodes=nodes)
        self.th_fx = create_feature_extractor(th_body, return_nodes=nodes)

        mid_in = int(l3_ch * 2)
        self.mid_fuse = nn.Sequential(
            nn.Conv2d(mid_in, l3_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(l3_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(l3_ch + d_deep + d_deep, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        rgb = x[:, :3]
        th = x[:, 3:4]
        o_r = self.rgb_fx(rgb)
        o_t = self.th_fx(th)
        r3, r_d = o_r["l3"], o_r["deep"]
        t3, t_d = o_t["l3"], o_t["deep"]
        cat = torch.cat([r3, t3], dim=1)
        f3_vec = torch.flatten(F.adaptive_avg_pool2d(self.mid_fuse(cat), (1, 1)), 1)
        z = torch.cat([f3_vec, r_d, t_d], dim=1)
        logits = self.head(z)
        if return_aux:
            return logits, {"mid_spatial_shape": tuple(r3.shape)}
        return logits


def _resnet_dim(backbone: str) -> int:
    b = str(backbone or "resnet18").lower()
    return 2048 if "50" in b else 512
