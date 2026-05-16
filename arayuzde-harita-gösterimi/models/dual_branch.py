"""
DualBranch Fusion model architectures.
Supports both standard fusion and gated fusion variants.
"""
import torch
import torch.nn as nn
from .backbones import make_feature_extractor, make_model


class DualBranchFusion(nn.Module):
    """Dual-branch RGB + thermal fusion (separate encoders, fused head)."""
    def __init__(self, backbone="resnet50", num_classes=2, pretrained=True, hidden=512):
        super().__init__()
        self.rgb_branch, d_rgb = make_feature_extractor(backbone, 3, pretrained=pretrained)
        self.th_branch, d_th = make_feature_extractor(backbone, 1, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(d_rgb + d_th, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        rgb, th = x[:, :3], x[:, 3:4]
        fr = self.rgb_branch(rgb).view(x.size(0), -1)
        ft = self.th_branch(th).view(x.size(0), -1)
        return self.head(torch.cat([fr, ft], dim=1))


class DualBranchGatedFusion(nn.Module):
    """Dual-branch with learned per-modality gating mechanism."""
    def __init__(self, backbone="resnet50", num_classes=2, pretrained=True, hidden=512, gate_hidden=128):
        super().__init__()
        self.rgb_branch, d_rgb = make_feature_extractor(backbone, 3, pretrained=pretrained)
        self.th_branch, d_th = make_feature_extractor(backbone, 1, pretrained=pretrained)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_rgb + d_th, gate_hidden), nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_rgb + d_th, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        rgb, th = x[:, :3], x[:, 3:4]
        fr = self.rgb_branch(rgb).view(x.size(0), -1)
        ft = self.th_branch(th).view(x.size(0), -1)
        z = torch.cat([fr, ft], dim=1)
        gate = self.gate_mlp(z)
        fr = fr * gate[:, 0:1]
        ft = ft * gate[:, 1:2]
        return self.head(torch.cat([fr, ft], dim=1))


def load_checkpoint(ckpt_path):
    """Load a DualBranch checkpoint and return (model, mode, device, threshold, temperature)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    
    backbone = ckpt.get("backbone", "resnet18")
    model_family = ckpt.get("model_family") or ckpt.get("arch")
    
    if model_family == "dual_branch_gated_fusion":
        model = DualBranchGatedFusion(backbone=backbone, num_classes=2, pretrained=False).to(device)
    elif model_family == "dual_branch_fusion":
        model = DualBranchFusion(backbone=backbone, num_classes=2, pretrained=False).to(device)
    else:
        in_ch = int(ckpt.get("in_ch", 4))
        model = make_model(backbone, in_ch, pretrained=False).to(device)
    
    model.load_state_dict(ckpt["state"])
    model.eval()
    
    thr = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    mode = ckpt.get("mode", "fusion")
    return model, mode, device, thr, temperature


def prep_rgb(frame_bgr, size=384):
    """Preprocess a BGR frame to normalized RGB tensor array (3, size, size)."""
    import cv2
    import numpy as np
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return arr, rgb


def prep_thermal(frame_bgr_or_gray, size=384):
    """Preprocess a thermal frame to normalized tensor array (1, size, size)."""
    import cv2
    import numpy as np
    if frame_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr_or_gray
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    finite_mask = np.isfinite(gray)
    fill = float(np.median(gray[finite_mask])) if finite_mask.any() else 0.0
    gray = np.where(finite_mask, gray, fill).astype(np.float32)
    lo = float(np.percentile(gray, 2.0))
    hi = float(np.percentile(gray, 98.0))
    if hi - lo < 1e-6:
        norm = np.zeros_like(gray, dtype=np.float32)
    else:
        norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm[None, ...], gray
