"""
Data utilities for indexing and dataset loading.

This package is intentionally lightweight: it provides
- dataset loading (`FlameDataset`)
- dataset splits
- index builder for FLAME-style directory layouts
"""

from .build_master_index import build_master_index  # noqa: F401
from .dataset import FlameDataset, read_rgb_pil, read_thermal_raw, thermal_to_norm01  # noqa: F401

__all__ = [
    "FlameDataset",
    "build_master_index",
    "read_rgb_pil",
    "read_thermal_raw",
    "thermal_to_norm01",
]
