from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Augmentation profiles
# -----------------------------------------------------------------------------
# Keep the model robust to the *same* perturbations exercised by the
# `robustness_eval` corruption variants. The eval-time corrupter (see
# ``src/training/robustness_eval.py``) injects RGB Gaussian noise sigma~0.038,
# thermal Gaussian noise sigma~0.042, RGB blur sigma=1.6, thermal shift/scale
# 1.14x and combined noise. Training augmentation must be a *superset* of those
# so the model sees similar distributions and does not collapse FPR under noise.
# Profiles are selected via the ``aug_strength`` constructor argument (CLI
# ``--aug_strength``). Backward-compat default = ``"default"``.

_BASE_DEFAULTS: dict = {
    # RGB photometric jitter
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.4,
    "p_jitter": 0.8,
    # RGB blur (Gaussian)
    "p_blur": 0.2,
    "blur_radius_min": 0.2,
    "blur_radius_max": 1.0,
    # RGB additive Gaussian noise (in [0,1] CHW space)
    "p_rgb_noise": 0.0,
    "sigma_rgb": 0.0,
    # Thermal noise
    "p_thermal_noise": 0.5,
    "sigma_thermal": 0.02,
    # Thermal contrast / mean shift (matches eval thermal_shift_scale)
    "p_thermal_shift_scale": 0.0,
    "thermal_scale_jitter": 0.0,  # multiplicative around 0.5
    "thermal_shift_jitter": 0.0,  # additive after re-centering
    # Combined-modality noise (mirrors rgb_thermal_combined_noise variant)
    "p_combined_noise": 0.0,
    "sigma_combined_rgb": 0.0,
    "sigma_combined_thermal": 0.0,
    # Random erasing on RGB
    "p_random_erase": 0.25,
}

_AUG_PROFILES: dict[str, dict] = {
    # No augmentation at all (sanity / debugging).
    "off": {
        **_BASE_DEFAULTS,
        "p_jitter": 0.0,
        "p_blur": 0.0,
        "p_rgb_noise": 0.0,
        "sigma_rgb": 0.0,
        "p_thermal_noise": 0.0,
        "sigma_thermal": 0.0,
        "p_thermal_shift_scale": 0.0,
        "p_combined_noise": 0.0,
        "p_random_erase": 0.0,
    },
    # Mild profile: same shape, lower intensities (good for tiny models).
    "light": {
        **_BASE_DEFAULTS,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "p_jitter": 0.6,
        "p_blur": 0.1,
        "p_rgb_noise": 0.2,
        "sigma_rgb": 0.015,
        "p_thermal_noise": 0.4,
        "sigma_thermal": 0.020,
        "p_random_erase": 0.15,
    },
    # Backward-compatible (legacy hardcoded values).
    "default": dict(_BASE_DEFAULTS),
    # Stronger augmentation: covers most of robustness_eval distribution.
    "strong": {
        **_BASE_DEFAULTS,
        "p_blur": 0.35,
        "blur_radius_max": 1.6,
        "p_rgb_noise": 0.5,
        "sigma_rgb": 0.030,
        "p_thermal_noise": 0.6,
        "sigma_thermal": 0.040,
        "p_thermal_shift_scale": 0.30,
        "thermal_scale_jitter": 0.10,
        "thermal_shift_jitter": 0.04,
        "p_combined_noise": 0.20,
        "sigma_combined_rgb": 0.025,
        "sigma_combined_thermal": 0.035,
    },
    # Strict superset of robustness_eval corruption ranges. Use when external
    # / drone no_fire FPR explodes under noise; expect tiny clean-recall hit.
    "match_eval": {
        **_BASE_DEFAULTS,
        "brightness": 0.45,
        "contrast": 0.45,
        "p_jitter": 0.85,
        "p_blur": 0.45,
        "blur_radius_min": 0.3,
        "blur_radius_max": 2.0,
        "p_rgb_noise": 0.55,
        "sigma_rgb": 0.045,
        "p_thermal_noise": 0.65,
        "sigma_thermal": 0.050,
        "p_thermal_shift_scale": 0.40,
        "thermal_scale_jitter": 0.14,
        "thermal_shift_jitter": 0.06,
        "p_combined_noise": 0.30,
        "sigma_combined_rgb": 0.030,
        "sigma_combined_thermal": 0.038,
        "p_random_erase": 0.30,
    },
}


def resolve_aug_profile(name: str | dict | None) -> dict:
    """Return a profile dict; unknown names fall back to ``default``."""
    if isinstance(name, dict):
        out = dict(_BASE_DEFAULTS)
        out.update(name)
        return out
    key = str(name or "default").strip().lower()
    return dict(_AUG_PROFILES.get(key, _AUG_PROFILES["default"]))


def _augment_rgb_pil(
    img: Image.Image,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    p_jitter: float = 0.8,
    p_blur: float = 0.2,
    blur_radius_min: float = 0.2,
    blur_radius_max: float = 1.0,
) -> Image.Image:
    """Train-only RGB photometric augmentations (no geometry change)."""
    if torch.rand(()).item() < p_jitter and brightness > 0:
        f = 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * float(brightness)
        img = ImageEnhance.Brightness(img).enhance(max(0.0, f))
    if torch.rand(()).item() < p_jitter and contrast > 0:
        f = 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * float(contrast)
        img = ImageEnhance.Contrast(img).enhance(max(0.0, f))
    if torch.rand(()).item() < p_jitter and saturation > 0:
        f = 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * float(saturation)
        img = ImageEnhance.Color(img).enhance(max(0.0, f))
    if torch.rand(()).item() < p_blur:
        rmin = max(0.0, float(blur_radius_min))
        rmax = max(rmin + 1e-3, float(blur_radius_max))
        radius = float(rmin + torch.rand(()).item() * (rmax - rmin))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def _maybe_random_erase_chw(rgb_chw: np.ndarray, p: float = 0.25) -> np.ndarray:
    """Train-only random erasing on RGB CHW array (in-place safe; thermal untouched)."""
    if p <= 0 or torch.rand(()).item() >= p:
        return rgb_chw
    c, h, w = rgb_chw.shape
    eh = int(h * (0.05 + torch.rand(()).item() * 0.20))
    ew = int(w * (0.05 + torch.rand(()).item() * 0.20))
    if eh < 1 or ew < 1 or eh >= h or ew >= w:
        return rgb_chw
    i = int(torch.randint(0, h - eh, (1,)).item())
    j = int(torch.randint(0, w - ew, (1,)).item())
    rgb_chw[:, i : i + eh, j : j + ew] = 0.0
    return rgb_chw


def _maybe_rgb_gaussian_noise_chw(
    rgb_chw: np.ndarray, sigma: float = 0.0, p: float = 0.0
) -> np.ndarray:
    """Train-only additive Gaussian noise on RGB tensor in [0,1]; mirrors eval rgb_gaussian_noise."""
    if sigma <= 0 or p <= 0 or torch.rand(()).item() >= p:
        return rgb_chw
    noise = (torch.randn(rgb_chw.shape).numpy() * float(sigma)).astype(np.float32)
    return np.clip(rgb_chw + noise, 0.0, 1.0).astype(np.float32)


def _maybe_thermal_noise(th_arr01: np.ndarray, sigma: float = 0.02, p: float = 0.5) -> np.ndarray:
    """Train-only Gaussian noise on thermal map already normalised to [0, 1]."""
    if sigma <= 0 or p <= 0 or torch.rand(()).item() >= p:
        return th_arr01
    noise = (torch.randn(th_arr01.shape).numpy() * float(sigma)).astype(np.float32)
    return np.clip(th_arr01 + noise, 0.0, 1.0).astype(np.float32)


def _maybe_thermal_shift_scale(
    th_arr01: np.ndarray,
    p: float = 0.0,
    scale_jitter: float = 0.0,
    shift_jitter: float = 0.0,
) -> np.ndarray:
    """Train-only thermal mean/contrast jitter; mirrors eval thermal_shift_scale.

    Applies ``(x - 0.5) * (1 + Δs) + 0.5 + Δm`` with random Δs∈[-scale_jitter, +scale_jitter]
    and Δm∈[-shift_jitter, +shift_jitter]. Output clipped to [0, 1].
    """
    if p <= 0 or torch.rand(()).item() >= p:
        return th_arr01
    ds = float((torch.rand(()).item() * 2.0 - 1.0) * float(scale_jitter))
    dm = float((torch.rand(()).item() * 2.0 - 1.0) * float(shift_jitter))
    out = (th_arr01.astype(np.float32) - 0.5) * (1.0 + ds) + 0.5 + dm
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _maybe_combined_noise(
    rgb_chw: np.ndarray,
    th_arr01: np.ndarray,
    p: float = 0.0,
    sigma_rgb: float = 0.0,
    sigma_th: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply correlated RGB+thermal Gaussian noise simultaneously (eval rgb_thermal_combined_noise)."""
    if p <= 0 or torch.rand(()).item() >= p:
        return rgb_chw, th_arr01
    if sigma_rgb > 0:
        nr = (torch.randn(rgb_chw.shape).numpy() * float(sigma_rgb)).astype(np.float32)
        rgb_chw = np.clip(rgb_chw + nr, 0.0, 1.0).astype(np.float32)
    if sigma_th > 0:
        nt = (torch.randn(th_arr01.shape).numpy() * float(sigma_th)).astype(np.float32)
        th_arr01 = np.clip(th_arr01 + nt, 0.0, 1.0).astype(np.float32)
    return rgb_chw, th_arr01


def _hflip_pil(img: Image.Image) -> Image.Image:
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _rotate_pil(img: Image.Image, degrees: float) -> Image.Image:
    # keep size, bilinear
    return img.rotate(float(degrees), resample=Image.BILINEAR, expand=False)


def _random_resized_crop_square(img: Image.Image, out_size: int, scale=(0.90, 1.00), ratio=(0.95, 1.05)) -> Image.Image:
    """
    Minimal replacement for torchvision RandomResizedCrop for square output.
    Uses torch RNG so it's deterministic under torch seeding.
    """
    w, h = img.size
    if w <= 1 or h <= 1:
        return img.resize((int(out_size), int(out_size)), resample=Image.BILINEAR)

    area = float(w * h)
    for _ in range(10):
        target_area = area * float((scale[0] + (scale[1] - scale[0]) * torch.rand(()).item()))
        aspect = float((ratio[0] + (ratio[1] - ratio[0]) * torch.rand(()).item()))
        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / max(1e-9, aspect)) ** 0.5))
        if 1 <= crop_w <= w and 1 <= crop_h <= h:
            i = int(torch.randint(0, h - crop_h + 1, (1,)).item())
            j = int(torch.randint(0, w - crop_w + 1, (1,)).item())
            cropped = img.crop((j, i, j + crop_w, i + crop_h))
            return cropped.resize((int(out_size), int(out_size)), resample=Image.BILINEAR)

    # Fallback: center crop then resize
    side = min(w, h)
    j = int((w - side) // 2)
    i = int((h - side) // 2)
    cropped = img.crop((j, i, j + side, i + side))
    return cropped.resize((int(out_size), int(out_size)), resample=Image.BILINEAR)


def read_rgb_pil(path: str | Path) -> Image.Image:
    p = Path(path)
    img = Image.open(p).convert("RGB")
    return img


def read_thermal_raw(path: str | Path) -> tuple[np.ndarray, str]:
    """
    Read thermal image into float32 array.

    Returns: (array, kind)
    - kind "uint16": raw 16-bit (often Celsius*100 or sensor units)
    - kind "float": already float-like
    """
    p = Path(path)
    img = Image.open(p)
    arr = np.array(img)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32), "uint16"
    return arr.astype(np.float32), "float"


def thermal_to_norm01(th: np.ndarray, kind: str = "uint16", norm: str = "percentile") -> np.ndarray:
    """
    Normalize thermal image to [0,1] robustly.

    Default percentile clamp is p2–p98 (a bit tighter than p1–p99) and NaN/inf
    values are replaced with the valid-pixel median before any statistic is taken
    so outliers do not dominate the scale. Keep this function in sync with
    `src/inference/preprocess.py::prep_thermal` so train/infer see the same
    thermal distribution.
    """
    x = np.asarray(th, dtype=np.float32)
    finite_mask = np.isfinite(x)
    if finite_mask.any():
        fill = float(np.median(x[finite_mask]))
    else:
        fill = 0.0
    x = np.where(finite_mask, x, fill).astype(np.float32)
    norm = (norm or "percentile").lower()

    # Fast path: most thermal16 TIFFs are uint16.
    if norm in ("uint16_div", "uint16"):
        if kind == "uint16":
            return np.clip(x / 65535.0, 0.0, 1.0).astype(np.float32)
        # Non-uint16 inputs fall back to min/max
        norm = "minmax"

    if norm == "minmax":
        if x.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # Robust default: p2–p98 percentiles (tighter than p1/p99 so the scale is
    # less sensitive to sensor spikes while still being robust to outliers).
    if x.size:
        lo = float(np.percentile(x, 2.0))
        hi = float(np.percentile(x, 98.0))
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _resize_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((int(size), int(size)), resample=Image.BILINEAR)


def _to_chw01_rgb(img: Image.Image) -> np.ndarray:
    arr = (np.asarray(img).astype(np.float32) / 255.0)
    return arr.transpose(2, 0, 1)


def _to_1hw01(th_norm01: np.ndarray) -> np.ndarray:
    if th_norm01.ndim == 3:
        th_norm01 = th_norm01[..., 0]
    return th_norm01[None, ...].astype(np.float32)


def _thermal_norm_to_gray2d(th_norm: np.ndarray) -> np.ndarray:
    """
    Thermal exports may be HxW (single band) or HxWxC (e.g. saved-as-RGB PNG).
    PIL mode L requires a 2D array.
    """
    x = np.asarray(th_norm, dtype=np.float32)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # Multi-band: average (common for accidental RGB copies of thermal)
        return np.mean(x, axis=-1)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected thermal map 2D after squeeze, got shape {th_norm.shape}")
    return x.astype(np.float32)


def _sync_geom_aug(rgb: Image.Image, th: Image.Image, out_size: int) -> tuple[Image.Image, Image.Image]:
    """
    Apply the SAME geometric augmentations to RGB and thermal.
    - horizontal flip
    - small rotation
    - random resized crop (resize jitter)
    """
    # Random horizontal flip
    if torch.rand(()) < 0.5:
        rgb = _hflip_pil(rgb)
        th = _hflip_pil(th)

    # Small rotation
    deg = float((torch.rand(()) * 10.0 - 5.0).item())  # [-5, +5]
    rgb = _rotate_pil(rgb, deg)
    th = _rotate_pil(th, deg)

    # Resize jitter via random resized crop (square -> square)
    rgb = _random_resized_crop_square(rgb, out_size=out_size, scale=(0.90, 1.00), ratio=(0.95, 1.05))
    th = _random_resized_crop_square(th, out_size=out_size, scale=(0.90, 1.00), ratio=(0.95, 1.05))
    return rgb, th


class FlameDataset(Dataset):
    """
    Minimal dataset wrapper used by `src/training/trainer.py`.

    Expected df columns:
    - `path_rgb` (str)
    - `path_th` or `path_thermal` (str) for thermal/fusion
    - `label` or `label_fire` (int 0/1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "fusion",
        size: int = 384,
        train: bool = False,
        thermal_norm: str = "percentile",
        aug_strength: str | dict | None = "default",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.mode = (mode or "fusion").lower()
        self.size = int(size)
        self.train = bool(train)
        self.thermal_norm = str(thermal_norm or "percentile")
        self.aug = resolve_aug_profile(aug_strength)

        if "label_fire" in self.df.columns:
            self.labels = pd.to_numeric(self.df["label_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
        else:
            self.labels = pd.to_numeric(self.df.get("label", 0), errors="coerce").fillna(0).astype(int).to_numpy()

        self.th_col = "path_th" if "path_th" in self.df.columns else ("path_thermal" if "path_thermal" in self.df.columns else None)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)]
        y = int(self.labels[int(idx)])

        rgb_path = str(r["path_rgb"]) if "path_rgb" in r.index else ""
        rgb_pil = read_rgb_pil(rgb_path)

        ap = self.aug
        if self.mode == "rgb":
            rgb_pil = _resize_pil(rgb_pil, self.size)
            if self.train:
                rgb_pil, _dummy = _sync_geom_aug(rgb_pil, rgb_pil.convert("L"), self.size)
                rgb_pil = _augment_rgb_pil(
                    rgb_pil,
                    brightness=float(ap["brightness"]),
                    contrast=float(ap["contrast"]),
                    saturation=float(ap["saturation"]),
                    p_jitter=float(ap["p_jitter"]),
                    p_blur=float(ap["p_blur"]),
                    blur_radius_min=float(ap["blur_radius_min"]),
                    blur_radius_max=float(ap["blur_radius_max"]),
                )
            x = _to_chw01_rgb(rgb_pil)
            if self.train:
                x = _maybe_rgb_gaussian_noise_chw(
                    x, sigma=float(ap["sigma_rgb"]), p=float(ap["p_rgb_noise"])
                )
                x = _maybe_random_erase_chw(x, p=float(ap["p_random_erase"]))
        else:
            if self.th_col is None:
                raise ValueError("Thermal path column missing (expected path_th or path_thermal).")
            th_path = str(r[self.th_col])
            th_raw, kind = read_thermal_raw(th_path)
            # Normalize thermal -> uint8 for sync geom augs (keeps pairing correct).
            th_norm = thermal_to_norm01(th_raw, kind=kind, norm=self.thermal_norm)
            th_norm = _thermal_norm_to_gray2d(th_norm)
            th_u8 = (np.clip(th_norm, 0, 1) * 255).astype(np.uint8)
            th_pil = Image.fromarray(th_u8, mode="L")

            # Apply SAME geometry to both (preserves RGB↔thermal alignment).
            rgb_pil = _resize_pil(rgb_pil, self.size)
            th_pil = _resize_pil(th_pil, self.size)
            if self.train:
                rgb_pil, th_pil = _sync_geom_aug(rgb_pil, th_pil, self.size)
                # RGB-only photometric perturbations: do NOT touch thermal.
                if self.mode == "fusion":
                    rgb_pil = _augment_rgb_pil(
                        rgb_pil,
                        brightness=float(ap["brightness"]),
                        contrast=float(ap["contrast"]),
                        saturation=float(ap["saturation"]),
                        p_jitter=float(ap["p_jitter"]),
                        p_blur=float(ap["p_blur"]),
                        blur_radius_min=float(ap["blur_radius_min"]),
                        blur_radius_max=float(ap["blur_radius_max"]),
                    )

            rgb = _to_chw01_rgb(rgb_pil)
            th_arr = (np.asarray(th_pil).astype(np.float32) / 255.0)
            if self.train:
                if self.mode == "fusion":
                    # Combined modality noise (mirrors eval rgb_thermal_combined_noise).
                    rgb, th_arr = _maybe_combined_noise(
                        rgb,
                        th_arr,
                        p=float(ap["p_combined_noise"]),
                        sigma_rgb=float(ap["sigma_combined_rgb"]),
                        sigma_th=float(ap["sigma_combined_thermal"]),
                    )
                    # Independent RGB Gaussian noise (eval rgb_gaussian_noise).
                    rgb = _maybe_rgb_gaussian_noise_chw(
                        rgb, sigma=float(ap["sigma_rgb"]), p=float(ap["p_rgb_noise"])
                    )
                    # RGB random erasing AFTER all photometric noise.
                    rgb = _maybe_random_erase_chw(rgb, p=float(ap["p_random_erase"]))
                # Independent thermal Gaussian noise (eval thermal_gaussian_noise).
                th_arr = _maybe_thermal_noise(
                    th_arr,
                    sigma=float(ap["sigma_thermal"]),
                    p=float(ap["p_thermal_noise"]),
                )
                # Thermal contrast/mean shift (eval thermal_shift_scale).
                th_arr = _maybe_thermal_shift_scale(
                    th_arr,
                    p=float(ap["p_thermal_shift_scale"]),
                    scale_jitter=float(ap["thermal_scale_jitter"]),
                    shift_jitter=float(ap["thermal_shift_jitter"]),
                )
            th = _to_1hw01(th_arr)
            if self.mode == "thermal":
                x = th
            else:
                x = np.concatenate([rgb, th], axis=0)

        # Avoid extra copies vs torch.tensor(np_array)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        return xt, yt
