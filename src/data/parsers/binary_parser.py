"""Parse FLAME binary subtree (CSV + disk layout)."""
from __future__ import annotations

from pathlib import Path

from .common import list_images, pair_by_key

try:
    from config import FLAME_BINARY_ROOT, FLAME_BINARY_CSV
except ImportError:
    FLAME_BINARY_ROOT = Path("data/flame3/binary")
    FLAME_BINARY_CSV = FLAME_BINARY_ROOT / "rgbt_multimodal_data.csv"


def _label_from_path(p: str) -> int:
    return 1 if "Fire" in p and "No_Fire" not in p else 0


def _scene_from_rel(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "binary_" + "_".join(parts[:2])
    return "binary_" + parts[0]


def parse_binary_from_csv(binary_root: Path, csv_path: Path) -> list[dict]:
    import pandas as pd

    rows: list[dict] = []
    if not csv_path.exists():
        print("[binary] CSV not found, skipped:", csv_path)
        return rows

    df = pd.read_csv(csv_path, header=None, names=["path"])
    paths = {p.strip() for p in df["path"].astype(str).tolist() if p.strip()}

    def is_rgb(p):
        return "_rgb/" in p or "_w/" in p

    def to_pair_path(p):
        if "_rgb/" in p:
            return p.replace("_rgb/", "_thermal/", 1)
        if "_thermal/" in p:
            return p.replace("_thermal/", "_rgb/", 1)
        if "_w/" in p:
            return p.replace("_w/", "_t/", 1)
        if "_t/" in p:
            return p.replace("_t/", "_w/", 1)
        return None

    added = 0
    for rel in paths:
        if not is_rgb(rel):
            continue
        pair_rel = to_pair_path(rel)
        if pair_rel is None or pair_rel not in paths:
            continue
        abs_rgb = binary_root / rel
        abs_th = binary_root / pair_rel
        scene = _scene_from_rel(rel)
        rows.append(
            {
                "source_dataset": "binary_fire",
                "scene_id": scene,
                "split_group": scene,
                "path_rgb": str(abs_rgb.resolve()),
                "path_thermal": str(abs_th.resolve()),
                "path_mask": "",
                "path_bbox": "",
                "label_fire": _label_from_path(rel),
                "label_quality": "weak",
                "sensor_mode": "fusion",
                "platform": "drone",
                "subset": "binary_csv",
                "legacy_key": rel.replace("_rgb/", "_").replace("_thermal/", "_"),
            }
        )
        added += 1
    print(f"[binary] from CSV: {len(paths)} paths -> {added} pairs")
    return rows


def parse_binary_from_disk(binary_root: Path, skip_rgb_paths: set[str]) -> list[dict]:
    binary_root = Path(binary_root)
    rows: list[dict] = []
    if not binary_root.exists():
        return rows

    candidates = [
        (binary_root / "train" / "Fire", 1),
        (binary_root / "train" / "No_Fire", 0),
        (binary_root / "Fire", 1),
        (binary_root / "No Fire", 0),
    ]

    def is_rgb_dir(d: Path) -> bool:
        n = d.name
        return "_rgb" in n or n.endswith("_w")

    def is_thermal_dir(d: Path) -> bool:
        n = d.name
        return "_thermal" in n or n.endswith("_t")

    def base_key(d: Path) -> str:
        n = d.name
        for s in ("_rgb", "_thermal", "_w", "_t"):
            n = n.replace(s, "")
        return n

    added = 0
    for cat_dir, label in candidates:
        if not cat_dir.exists():
            continue
        subdirs = [p for p in cat_dir.iterdir() if p.is_dir()]
        rgb_dirs = {base_key(d): d for d in subdirs if is_rgb_dir(d)}
        th_dirs = {base_key(d): d for d in subdirs if is_thermal_dir(d)}
        for base in rgb_dirs:
            if base not in th_dirs:
                continue
            r_dir, t_dir = rgb_dirs[base], th_dirs[base]
            rgb_files = list_images(r_dir)
            th_files = list_images(t_dir)
            pairs = pair_by_key(rgb_files, th_files)
            scene_base = f"binary_disk_{cat_dir.name}_{base}"
            for r, t, k in pairs:
                path_rgb = str(r)
                if path_rgb in skip_rgb_paths:
                    continue
                skip_rgb_paths.add(path_rgb)
                rows.append(
                    {
                        "source_dataset": "binary_fire",
                        "scene_id": f"{scene_base}_{k}",
                        "split_group": scene_base,
                        "path_rgb": path_rgb,
                        "path_thermal": str(t),
                        "path_mask": "",
                        "path_bbox": "",
                        "label_fire": int(label),
                        "label_quality": "weak",
                        "sensor_mode": "fusion",
                        "platform": "drone",
                        "subset": "binary_disk",
                        "legacy_key": f"{cat_dir.name}_{base}_{k}",
                    }
                )
                added += 1
    if added:
        print(f"[binary] from disk: +{added} pairs")
    return rows


def parse_binary_folders(binary_root: Path | None = None, csv_path: Path | None = None) -> list[dict]:
    binary_root = Path(binary_root or FLAME_BINARY_ROOT)
    csv_path = Path(csv_path or FLAME_BINARY_CSV)
    rows = parse_binary_from_csv(binary_root, csv_path)
    skip = {r["path_rgb"] for r in rows}
    rows.extend(parse_binary_from_disk(binary_root, skip))
    return rows
