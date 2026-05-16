# src/model_loader.py

import torch

from models.cls.fusion_variants import DualBranchGatedFusion


def load_dual_branch_model(
    checkpoint_path="outputs/checkpoints/dual_branch.pt",
    device=None
):
    """
    dual_branch_gated_fusion checkpoint'ini yükler.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)

    model_family = ckpt.get("model_family")
    backbone = ckpt.get("backbone", "resnet50")
    threshold = ckpt.get("threshold_recommended", ckpt.get("threshold", 0.5))
    input_size = ckpt.get("input_size", 384)
    temperature = ckpt.get("temperature", 1.0)

    if model_family != "dual_branch_gated_fusion":
        raise ValueError(f"Unsupported model_family: {model_family}")

    model = DualBranchGatedFusion(
        backbone=backbone,
        num_classes=2,
        pretrained=False
    )

    state_dict = ckpt["state"]
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, {
        "device": device,
        "threshold": threshold,
        "input_size": input_size,
        "temperature": temperature,
        "class_mapping": ckpt.get("class_mapping"),
        "model_family": model_family,
        "backbone": backbone
    }


def predict_fire_probability(model, input_tensor, temperature=1.0):
    """
    Modelden fire probability döndürür.
    input_tensor shape:
        [1, 4, H, W]
    """

    with torch.no_grad():
        logits = model(input_tensor)

        if temperature and temperature > 0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=1)

        fire_prob = probs[0, 1].item()

    return fire_prob