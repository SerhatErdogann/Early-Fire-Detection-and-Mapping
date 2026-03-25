import torch
from pathlib import Path

try:
    from src.models.backbones import make_model
    from src.models.cls.dual_branch_fusion import DualBranchFusion
except ImportError:
    from ..models.backbones import make_model
    from ..models.cls.dual_branch_fusion import DualBranchFusion


def load_checkpoint(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    in_ch = int(ckpt["in_ch"])
    backbone = ckpt.get("backbone", "resnet18")
    model_family = ckpt.get("model_family") or ckpt.get("arch")
    if model_family == "dual_branch_fusion":
        model = DualBranchFusion(backbone=backbone, num_classes=2, pretrained=False).to(device)
    else:
        model = make_model(backbone, in_ch, pretrained=False).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    thr = float(ckpt.get("threshold", 0.5))
    temperature = float(ckpt.get("temperature", 1.0))
    mode = ckpt.get("mode", "fusion")
    return model, mode, device, thr, temperature
