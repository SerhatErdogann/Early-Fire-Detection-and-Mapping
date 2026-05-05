import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossCE(nn.Module):
    """Focal loss for imbalanced fire/no-fire classification."""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class LabelSmoothingCE(nn.Module):
    def __init__(self, weight=None, smoothing: float = 0.05, num_classes: int = 2):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.smoothing = float(smoothing)
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = torch.sum(-true_dist * log_probs, dim=1)
        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w
        return loss.mean()


class ClassBalancedFocalCE(nn.Module):
    """Focal loss with optional per-class effective number weights (CB loss style)."""

    def __init__(
        self,
        counts_per_class: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        *,
        manual_class_gain: tuple[float, float] | None = None,
        device_hint: torch.device | None = None,
    ):
        super().__init__()
        device = counts_per_class.device if isinstance(counts_per_class, torch.Tensor) else (
            device_hint or torch.device("cpu")
        )
        eff_num = 1.0 - torch.pow(beta, counts_per_class.to(device))
        w = (1.0 - beta) / (eff_num + 1e-12)
        w = w / w.sum() * len(counts_per_class)
        if manual_class_gain is not None:
            g0, g1 = float(manual_class_gain[0]), float(manual_class_gain[1])
            if g0 > 0 and g1 > 0:
                sc = torch.tensor([g0, g1], dtype=torch.float32, device=device)
                w = (w.float() * sc) / ((w.float() * sc).sum() + 1e-12) * float(len(counts_per_class))
        self.register_buffer("weight", w.float())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(
    loss_name: str,
    class_weights: torch.Tensor | None,
    device: torch.device,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    class_counts: torch.Tensor | None = None,
    manual_class_gain: tuple[float, float] | None = None,
):
    name = (loss_name or "focal").lower()
    w = class_weights
    if name in ("sampler_ce", "weighted_ce", "ce"):
        return nn.CrossEntropyLoss(weight=w)
    if name in ("label_smoothing_ce", "ls_ce"):
        return LabelSmoothingCE(weight=w, smoothing=label_smoothing or 0.05)
    if name in ("class_balanced_focal", "cb_focal"):
        if class_counts is None:
            class_counts = torch.tensor([1.0, 1.0], device=device)
        return ClassBalancedFocalCE(
            class_counts.to(device),
            gamma=focal_gamma,
            manual_class_gain=manual_class_gain,
            device_hint=device,
        )
    return FocalLossCE(weight=w, gamma=focal_gamma)
