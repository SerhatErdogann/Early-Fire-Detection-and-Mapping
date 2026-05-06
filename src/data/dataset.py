from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


def _augment_rgb_pil(
    img: Image.Image,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    p_jitter: float = 0.8,
    p_blur: float = 0.2,
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
        radius = float(0.2 + torch.rand(()).item() * 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def _maybe_random_erase_chw(rgb_chw: np.ndarray, p: float = 0.25) -> np.ndarray:
    """Train-only random erasing on RGB CHW array (in-place safe; thermal untouched)."""
    if torch.rand(()).item() >= p:
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


def _maybe_thermal_noise(th_arr01: np.ndarray, sigma: float = 0.02, p: float = 0.5) -> np.ndarray:
    """Train-only Gaussian noise on thermal map already normalised to [0, 1]."""
    if sigma <= 0 or torch.rand(()).item() >= p:
        return th_arr01
    noise = (torch.randn(th_arr01.shape).numpy() * float(sigma)).astype(np.float32)
    return np.clip(th_arr01 + noise, 0.0, 1.0).astype(np.float32)


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
    ):
        self.df = df.reset_index(drop=True).copy()
        self.mode = (mode or "fusion").lower()
        self.size = int(size)
        self.train = bool(train)
        self.thermal_norm = str(thermal_norm or "percentile")

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

        if self.mode == "rgb":
            rgb_pil = _resize_pil(rgb_pil, self.size)
            if self.train:
                rgb_pil, _dummy = _sync_geom_aug(rgb_pil, rgb_pil.convert("L"), self.size)
                rgb_pil = _augment_rgb_pil(rgb_pil)
            x = _to_chw01_rgb(rgb_pil)
            if self.train:
                x = _maybe_random_erase_chw(x)
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
                    rgb_pil = _augment_rgb_pil(rgb_pil)

            rgb = _to_chw01_rgb(rgb_pil)
            th_arr = (np.asarray(th_pil).astype(np.float32) / 255.0)
            if self.train:
                # RGB random erasing applied AFTER geometry, only for fusion;
                # thermal stays clean to preserve modality consistency.
                if self.mode == "fusion":
                    rgb = _maybe_random_erase_chw(rgb)
                # Light Gaussian noise on thermal map only.
                th_arr = _maybe_thermal_noise(th_arr, sigma=0.02)
            th = _to_1hw01(th_arr)
            if self.mode == "thermal":
                x = th
            else:
                x = np.concatenate([rgb, th], axis=0)

        # Avoid extra copies vs torch.tensor(np_array)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        return xt, yt
