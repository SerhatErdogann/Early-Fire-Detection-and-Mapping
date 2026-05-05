from __future__ import annotations

"""
Lightweight U-Net for fire segmentation (baseline scaffold).
Train with mask supervision when ``path_mask`` is populated in the master index.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bot = _ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)

    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        binary = (prob >= 0.5).float()
        return prob, binary
