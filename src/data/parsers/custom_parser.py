"""
Scan extra data roots (e.g. extracted data.rar) for common fire / no_fire layouts.
Produces weak labels from folder names only.
"""
from __future__ import annotations

import re
from pathlib import Path

from .common import list_images, pair_by_key, make_key

# Typical folder name hints
_FIRE_NAMES = re.compile(r"fire|yangin|alev|flame", re.I)
_NOFIRE_NAMES = re.compile(r"no_?fire|nofire|neg|background|normal", re.I)


def _infer_label_from_parts(parts: list[str]) -> int | None:
    joined = "/".join(parts).lower()
    if _NOFIRE_NAMES.search(joined) and not _FIRE_NAMES.search(joined.replace("no_fire", "").replace("nofire", "")):
        if "no" in joined and "fire" in joined:
            return 0
    if _FIRE_NAMES.search(joined):
        if _NOFIRE_NAMES.search(joined):
            return 0
        return 1
    if "no_fire" in joined or "nofire" in joined or "no fire" in joined:
        return 0
    return None


def _find_paired_subdirs(parent: Path) -> list[tuple[Path, Path, str]]:
    """If parent has rgb-like and thermal-like subfolders, return (rgb_dir, th_dir, base_id)."""
    if not parent.is_dir():
        return []
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return []
    rgb_candidates = [d for d in subdirs if re.search(r"rgb|visible|_w$|color", d.name, re.I)]
    th_candidates = [d for d in subdirs if re.search(r"thermal|ir|infra|_t$|th$", d.name, re.I)]
    out = []
    for rd in rgb_candidates:
        for td in th_candidates:
            base = make_key(Path(rd.name + "_" + td.name))
            out.append((rd, td, base))
    return out


def parse_custom_data_rar(
    data_root: Path,
    source_name: str = "data_rar_x",
    max_depth: int = 6,
) -> list[dict]:
    """
    Walk data_root up to max_depth; detect:
    - directories named fire/no_fire (or binary/fire) with paired RGB+thermal subfolders
    - single-modality folders (skipped for fusion master; could extend later)
    """
    data_root = Path(data_root)
    rows: list[dict] = []
    if not data_root.exists():
        print("[custom] root missing:", data_root)
        return rows

    seen_rgb: set[str] = set()

    for dirpath, dirnames, _filenames in _walk(data_root, max_depth):
        p = Path(dirpath)
        rel_parts = p.relative_to(data_root).parts if p != data_root else ()
        if len(rel_parts) > max_depth:
            dirnames[:] = []
            continue

        label = _infer_label_from_parts(list(rel_parts) + [p.name])
        pairs = _find_paired_subdirs(p)
        if label is None or not pairs:
            continue

        for rgb_dir, th_dir, base in pairs:
            rgb_files = list_images(rgb_dir)
            th_files = list_images(th_dir)
            prs = pair_by_key(rgb_files, th_files)
            scene_root = f"{source_name}_{'_'.join(rel_parts[-3:])}" if rel_parts else source_name
            for r, t, k in prs:
                pr = str(r.resolve())
                if pr in seen_rgb:
                    continue
                seen_rgb.add(pr)
                rows.append(
                    {
                        "source_dataset": source_name,
                        "scene_id": f"{scene_root}_{k}",
                        "split_group": scene_root,
                        "path_rgb": pr,
                        "path_thermal": str(t.resolve()),
                        "path_mask": "",
                        "path_bbox": "",
                        "label_fire": int(label),
                        "label_quality": "weak",
                        "sensor_mode": "fusion",
                        "platform": "unknown",
                        "subset": "custom_scan",
                        "legacy_key": k,
                    }
                )

    print(f"[custom] {data_root} -> {len(rows)} pairs (heuristic)")
    return rows


def _walk(root: Path, max_depth: int):
    """Fallback os.walk with depth limit."""
    import os

    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        try:
            depth = len(p.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames, filenames
