"""Shared helpers for dataset index parsers."""
from __future__ import annotations

from pathlib import Path

try:
    from config import IMG_EXTENSIONS
except ImportError:
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])


def make_key(p: Path) -> str:
    stem = p.stem
    if stem.isdigit():
        return stem
    return stem.lower()


def pair_by_key(rgb_files, th_files):
    th_map = {}
    for t in th_files:
        k = make_key(t)
        if k not in th_map:
            th_map[k] = t
    pairs = []
    for r in rgb_files:
        k = make_key(r)
        t = th_map.get(k)
        if t is not None:
            pairs.append((r, t, k))
    return pairs
