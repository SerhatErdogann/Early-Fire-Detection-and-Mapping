"""Parse FLAME-style RGB + thermal paired folders."""
from __future__ import annotations

from pathlib import Path

from .common import list_images, pair_by_key

try:
    from config import FLAME_ROOT
except ImportError:
    FLAME_ROOT = Path("data/flame3")


def _row(
    path_rgb: Path,
    path_thermal: Path,
    key: str,
    label_fire: int,
    subset: str,
    label_quality: str,
    scene_prefix: str,
):
    sid = f"{scene_prefix}_{key}"
    return {
        "source_dataset": "flame",
        "scene_id": sid,
        "split_group": sid,
        "path_rgb": str(path_rgb.resolve()),
        "path_thermal": str(path_thermal.resolve()),
        "path_mask": "",
        "path_bbox": "",
        "label_fire": int(label_fire),
        "label_quality": label_quality,
        "sensor_mode": "fusion",
        "platform": "drone",
        "subset": subset,
        "legacy_key": key,
    }


def parse_flame(flame_root: Path | None = None) -> list[dict]:
    flame_root = Path(flame_root or FLAME_ROOT)
    rows: list[dict] = []
    for label_name, label_val, quality in (
        ("Fire", 1, "gold"),
        ("No Fire", 0, "gold"),
    ):
        rgb_dir = flame_root / label_name / "RGB" / "Corrected FOV"
        th_dir = flame_root / label_name / "Thermal" / "Celsius TIFF"
        rgb_files = list_images(rgb_dir)
        th_files = list_images(th_dir)
        print(f"[flame] {label_name} RGB: {len(rgb_files)} | Thermal: {len(th_files)}")
        pairs = pair_by_key(rgb_files, th_files)
        print(f"[flame] {label_name} paired: {len(pairs)}")
        prefix = f"flame3_{label_name.replace(' ', '_')}"
        for r, t, k in pairs:
            rows.append(_row(r, t, k, label_val, "flame_main", quality, prefix))

    extra_root = flame_root / "extra"
    if extra_root.exists():
        rgb_dir = extra_root / "RGB"
        th_dir = extra_root / "Thermal"
        rgb_files = list_images(rgb_dir)
        th_files = list_images(th_dir)
        print(f"[flame] extra RGB: {len(rgb_files)} | Thermal: {len(th_files)}")
        pairs = pair_by_key(rgb_files, th_files)
        print(f"[flame] extra paired: {len(pairs)}")
        for r, t, k in pairs:
            rows.append(_row(r, t, k, 0, "flame_extra", "silver", "flame3_extra"))
    else:
        print("[flame] extra folder not found, skipped:", extra_root)

    return rows
