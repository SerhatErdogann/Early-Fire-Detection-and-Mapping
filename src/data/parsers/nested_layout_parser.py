"""Scan flame3/binary, flame3/dataset, etc.: each has fire|no fire folders with rgb + thermal."""
from __future__ import annotations

from pathlib import Path

from .common import list_images, pair_by_key

try:
    from config import FLAME_ROOT, FLAME_NESTED_SCAN
except ImportError:
    FLAME_ROOT = Path("data/flame3")
    FLAME_NESTED_SCAN = ["binary", "dataset"]


def _label_from_split_folder(name: str) -> int | None:
    n = name.strip().lower().replace("-", "_")
    n2 = n.replace("_", " ").strip()
    compact = n2.replace(" ", "")
    if compact == "nofire" or n2 == "no fire" or n2.startswith("no fire "):
        return 0
    if n.startswith("no_fire") or n2.startswith("no_fire"):
        return 0
    if n2 == "fire":
        return 1
    return None


def _resolve_rgb_thermal_dirs(label_dir: Path) -> tuple[Path, Path] | None:
    """Return folders that directly contain image files (or FLAME-style nested leaves)."""
    rgb_cov = label_dir / "RGB" / "Corrected FOV"
    th_ct = label_dir / "Thermal" / "Celsius TIFF"
    if rgb_cov.is_dir() and th_ct.is_dir():
        return rgb_cov, th_ct

    rgb_top = label_dir / "RGB"
    th_top = label_dir / "Thermal"
    if rgb_top.is_dir() and th_top.is_dir():
        if list_images(rgb_top):
            return rgb_top, th_top
        cov = rgb_top / "Corrected FOV"
        cth = th_top / "Celsius TIFF"
        if cov.is_dir() and cth.is_dir():
            return cov, cth

    by_lower: dict[str, Path] = {}
    for c in label_dir.iterdir():
        if c.is_dir():
            by_lower.setdefault(c.name.lower(), c)
    if "rgb" in by_lower and "thermal" in by_lower:
        return by_lower["rgb"], by_lower["thermal"]

    dirs = [c for c in label_dir.iterdir() if c.is_dir()]
    rgb_cands = [
        d
        for d in dirs
        if "rgb" in d.name.lower() and "thermal" not in d.name.lower()
    ]
    th_cands = [
        d
        for d in dirs
        if "thermal" in d.name.lower() or d.name.lower() in ("ir", "t")
    ]
    if len(rgb_cands) == 1 and len(th_cands) == 1:
        return rgb_cands[0], th_cands[0]

    return None


def _looks_like_binary_scene_rgb_thermal_folders(label_dir: Path) -> bool:
    """Same layout as parse_binary_from_disk: Fire/No_Fire with sibling *_rgb and *_thermal dirs."""
    subs = [p for p in label_dir.iterdir() if p.is_dir()]
    if len(subs) < 2:
        return False

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

    rgb_bases = {base_key(d) for d in subs if is_rgb_dir(d)}
    th_bases = {base_key(d) for d in subs if is_thermal_dir(d)}
    return bool(rgb_bases & th_bases)


def _scan_base_dir(flame_root: Path, rel: str) -> Path | None:
    """Resolve config name to an existing child of flame_root (case-insensitive fallback)."""
    p = flame_root / rel
    if p.is_dir():
        return p
    if not flame_root.is_dir():
        return None
    target = rel.casefold()
    for c in flame_root.iterdir():
        if c.is_dir() and c.name.casefold() == target:
            return c
    return None


def parse_flame_nested_subtrees(
    flame_root: Path | None = None,
    subtree_names: list[str] | None = None,
) -> list[dict]:
    flame_root = Path(flame_root or FLAME_ROOT)
    names = subtree_names if subtree_names is not None else list(FLAME_NESTED_SCAN or [])
    rows: list[dict] = []
    for rel in names:
        rel = str(rel).strip()
        if not rel:
            continue
        base = _scan_base_dir(flame_root, rel)
        if base is None:
            print(f"[nested] skip (missing): {flame_root / rel}")
            continue
        added_here = 0
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            lab = _label_from_split_folder(child.name)
            if lab is None:
                continue
            resolved = _resolve_rgb_thermal_dirs(child)
            scene_prefix = f"{rel}_{child.name.replace(' ', '_')}"
            if resolved is None:
                flat = list_images(child)
                if flat:
                    print(
                        f"[nested] {base.name}/{child.name}: flat folder ({len(flat)} imgs) "
                        f"-> thermal path = rgb path (grayscale)"
                    )
                    for p in flat:
                        k = p.stem.lower()
                        ap = str(p.resolve())
                        rows.append(
                            {
                                "source_dataset": "flame_nested",
                                "scene_id": f"{scene_prefix}_{k}",
                                "split_group": scene_prefix,
                                "path_rgb": ap,
                                "path_thermal": ap,
                                "path_mask": "",
                                "path_bbox": "",
                                "label_fire": int(lab),
                                "label_quality": "silver",
                                "sensor_mode": "fusion",
                                "platform": "drone",
                                "subset": f"nested_{rel}_rgb_only",
                                "legacy_key": f"{rel}_{child.name}_{k}",
                            }
                        )
                        added_here += 1
                elif _looks_like_binary_scene_rgb_thermal_folders(child):
                    print(
                        f"[nested] {base.name}/{child.name}: *_rgb / *_thermal sahne klasörleri — "
                        "zaten [binary] disk+CSV parser ile indekslenir; nested tekrar eklemez."
                    )
                else:
                    print(f"[nested] skip (no rgb/thermal, flat empty): {child}")
                continue
            rgb_dir, th_dir = resolved
            rgb_files = list_images(rgb_dir)
            th_files = list_images(th_dir)
            pairs = pair_by_key(rgb_files, th_files)
            print(
                f"[nested] {base.name}/{child.name} RGB: {len(rgb_files)} | "
                f"Thermal: {len(th_files)} | paired: {len(pairs)}"
            )
            for r, t, k in pairs:
                rows.append(
                    {
                        "source_dataset": "flame_nested",
                        "scene_id": f"{scene_prefix}_{k}",
                        "split_group": scene_prefix,
                        "path_rgb": str(r.resolve()),
                        "path_thermal": str(t.resolve()),
                        "path_mask": "",
                        "path_bbox": "",
                        "label_fire": int(lab),
                        "label_quality": "gold",
                        "sensor_mode": "fusion",
                        "platform": "drone",
                        "subset": f"nested_{rel}",
                        "legacy_key": f"{rel}_{child.name}_{k}",
                    }
                )
                added_here += 1
        if added_here:
            print(f"[nested] subtree {rel!r}: +{added_here} pairs")
    return rows
