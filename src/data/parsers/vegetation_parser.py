"""Placeholder for future vegetation / fuel-type datasets."""
from __future__ import annotations

from pathlib import Path


def parse_future_vegetation(data_root: Path | None = None) -> list[dict]:
    """
    When a vegetation dataset is added, implement discovery here and return
    master-schema rows (path_rgb, path_thermal optional, vegetation_type, etc.).
    """
    _ = data_root
    return []
