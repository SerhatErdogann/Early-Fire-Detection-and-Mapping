from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import (
        FLAME_EMBEDDED_CSV,
        FLAME_INDEX_CSV,
        MASTER_INDEX_PARQUET,
        OUTPUTS_DIR,
        FLAME_VIDEO_FRAMES_ROOT,
        BINARY_ROOT,
        FLAME_ROOT,
        DATA_ROOT,
    )
except Exception:  # pragma: no cover
    FLAME_ROOT = Path("data/flame3")
    BINARY_ROOT = Path("data/binary")
    OUTPUTS_DIR = Path("outputs")
    FLAME_INDEX_CSV = OUTPUTS_DIR / "flame_index.csv"
    MASTER_INDEX_PARQUET = Path("data/master_index.parquet")
    FLAME_EMBEDDED_CSV = FLAME_ROOT / "binary" / "rgbt_multimodal_data.csv"
    FLAME_VIDEO_FRAMES_ROOT = Path("data/flame_video_frames")
    DATA_ROOT = Path("data")


def _resolve_root_env_cli_default(
    env_key: str,
    cli: str | Path | None,
    default_path: Path,
) -> tuple[Path | None, str]:
    """Preference order: env > CLI > canonical default."""
    raw = (os.environ.get(env_key) or "").strip()
    if raw:
        return Path(raw).expanduser(), env_key + "(env)"
    if cli is not None and str(cli).strip():
        return Path(str(cli).strip()).expanduser(), "cli"
    return Path(default_path), "fallback"


def _resolve_cart_layout_base(user_point: Path) -> Path | None:
    """Return a directory that contains ``color/`` and ``thermal16/``.

    Tries ``user_point`` then ``user_point/cart`` (typical Kaggle layouts).
    """
    r = Path(user_point).expanduser().resolve()
    for base in (r, r / "cart"):
        c, th = base / "color", base / "thermal16"
        if c.is_dir() and th.is_dir():
            return base
    return None


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Modality suffixes that should be stripped from filename stems so RGB↔Thermal
# pairing succeeds even when the two trees use different naming conventions
# (e.g. ``frame_001_rgb.jpg`` ↔ ``frame_001_thermal.tif``). Order matters:
# longer suffixes first so we don't strip ``_t`` from ``_thermal`` early.
_MODALITY_STEM_SUFFIXES = (
    "_thermal", "_visible", "_color", "_lwir",
    "_rgb", "_ir", "_t", "_w",
)


def _normalize_stem(p: Path) -> str:
    """Lower-cased stem with one trailing modality suffix stripped (if any)."""
    s = p.stem.lower()
    for suf in _MODALITY_STEM_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


_CART_DUAL_MOD_TAIL = re.compile(r"_(?:eo|thermal)-(\d+)$", re.IGNORECASE)


def _cart_pair_stem_key(p: Path) -> str:
    """
    CART naming: EO uses ``*_eo-<frame>``, thermal uses ``*_thermal-<frame>``.
    Map both to ``<prefix>_<frame>`` so paired frames match.
    Falls back to :func:`_normalize_stem` when the pattern doesn't apply.
    """
    s = _normalize_stem(p)
    m = _CART_DUAL_MOD_TAIL.search(s)
    if m:
        return (s[: m.start()] + "_" + str(m.group(1))).lower()
    return s


def _files_cart_side(root: Path) -> tuple[dict[str, Path], dict]:
    """Like `_files_by_stem_with_stats` but keys via :func:`_cart_pair_stem_key`."""
    out: dict[str, Path] = {}
    collisions = 0
    total = 0
    if not root.exists():
        return out, {"collisions": 0, "total": 0}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            total += 1
            key = _cart_pair_stem_key(p)
            if key in out:
                collisions += 1
                continue
            out[key] = p
    return out, {"collisions": collisions, "total": total}


# Trailing frame-number pattern, used by the leakage guard to detect groups of
# frames that share a sequence prefix (e.g. ``frame_test_mm_fire_0001``).
_FRAME_TAIL_RE = re.compile(r"[-_]?\d+$")

# Sources where the entire on-disk dataset is one continuous video and we
# explicitly accept block-level (chunked) splitting across train/val/test.
# For these sources the leakage guard's "video stem" heuristic
# (parent_path + stripped_stem) is too coarse — it would collapse every frame
# into a single video and forbid any 3-way split. We mitigate real leakage by
# (a) keeping consecutive frames in the same chunk and (b) inserting a small
# frame gap at every chunk boundary (see ``_scan_pairs_two_trees``).
_BLOCK_SPLIT_OPT_IN_SOURCES: set[str] = {"flame3"}

# Authoritative splits (never remapped by global rebalance fallback).
_SPLIT_LOCKED_SOURCES = frozenset({"binary_root"})


def _infer_official_split_from_path(p: str) -> str | None:
    """
    Heuristic: if dataset already encodes train/val/test in folder names, preserve it.
    We keep this conservative to avoid mis-detecting arbitrary 'train' substrings.
    """
    s = str(p).replace("\\", "/").lower()
    toks = [t for t in s.split("/") if t]
    keyset = set(toks)
    # common variants
    if "train" in keyset or "training" in keyset:
        return "train"
    if "val" in keyset or "valid" in keyset or "validation" in keyset:
        return "val"
    if "test" in keyset or "testing" in keyset:
        return "test"
    return None


def _extract_video_pair_id(key_or_path: str) -> str:
    """
    flame_video_nofire pair id, derived from the file stem (e.g.
    ``pair1_000001`` -> ``pair1``).

    Robust to both the raw filename stem AND a dataset-prefixed key like
    ``flame_video_nofire_pair1_000001``: in the latter case we strip the
    ``flame_video_nofire_`` prefix first.
    """
    s = str(key_or_path)
    # If a path is passed, take the stem.
    if "/" in s or "\\" in s:
        s = Path(s).stem
    s = s.lower()
    prefix = "flame_video_nofire_"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _partition_flame_video_pairs(
    pair_ids: list[str],
    row_counts: dict[str, int],
    *,
    target_test_frames: int = 420,
    max_test_frame_frac: float = 0.52,
) -> tuple[set[str], set[str], set[str]]:
    """Split ``flame_video_nofire`` **pair ids** across train/val/test.

    All rows in a pair share one split (no group leakage). The policy gives
    **train priority** so the model is exposed to drone no-fire footage during
    fitting; otherwise val / test FPR collapses against a domain it has never
    seen (the previous policy could leave train with zero pairs).

    Adaptive behaviour by available pair count:

    - ``n_pairs == 1`` → all to **train** (val/test fall back to other no-fire
      sources such as binary_root and flame3).
    - ``n_pairs == 2`` → larger clip to **train**, smaller clip to **test** (keep val empty;
      test gains dedicated ``flame_video_nofire`` no_fire only when ≥2 clips exist).
    - ``n_pairs >= 3`` → standard 3-way:
        * ``train`` always receives the largest pair (model sees this domain);
        * ``test`` is filled with a frame budget (greedy, largest pairs first
          from the remaining pool) up to ``target_test_frames`` and capped at
          ``max_test_frame_frac`` of total frames so it cannot dominate;
        * at least one pair remains for ``val``;
        * leftover pairs split ~82/18 train / val by clip count.
    """
    pairs = sorted(set(pair_ids))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return set(), set(), set()

    if n_pairs == 1:
        return set(pairs), set(), set()

    # Deterministic ordering: largest pair first.
    ordered = sorted(pairs, key=lambda pid: (-int(row_counts.get(pid, 0)), str(pid)))

    if n_pairs == 2:
        # Second-smallest clip → test when possible so no_fire footage appears in evaluation.
        return {ordered[0]}, set(), {ordered[1]}

    # n_pairs >= 3. Always reserve the biggest pair for train.
    train_seed = ordered[0]
    test_pool = ordered[1:]

    total_frames = int(sum(int(row_counts.get(p, 0)) for p in pairs))
    if total_frames <= 0:
        return set(pairs), set(), set()

    cap = min(max(1, int(total_frames * max_test_frame_frac + 1e-9)), total_frames)
    want = min(int(target_test_frames), cap, total_frames)

    p_test: set[str] = set()
    got = 0
    for pid in test_pool:
        # Stop before consuming the last remaining pair so val gets at least one.
        if got >= want or len(p_test) >= len(test_pool) - 1:
            break
        p_test.add(pid)
        got += int(row_counts.get(pid, 0))
    if want >= 1 and not p_test and len(test_pool) >= 2:
        # Guarantee at least one test pair when caller asked for any test frames.
        p_test.add(test_pool[0])

    rest = [p for p in test_pool if p not in p_test]
    if not rest:
        return {train_seed}, set(), p_test
    if len(rest) == 1:
        return {train_seed}, {rest[0]}, p_test

    n_va = max(1, int(round(len(rest) * 0.18)))
    n_va = min(n_va, len(rest) - 1)
    p_val = set(rest[:n_va])
    p_train_extra = set(rest[n_va:])
    p_train = {train_seed} | p_train_extra
    return p_train, p_val, p_test


def _stratified_group_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    stratify_key: str = "label",
    group_col: str = "split_group",
) -> pd.Series:
    """
    Assign split per group (no leakage) while approximately stratifying.
    Returns a Series of 'train'/'val'/'test' aligned to df.index.
    """
    r_tr, r_va, r_te = ratios
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    if df.empty:
        return pd.Series([], index=df.index, dtype=str)

    d = df.copy()
    if group_col not in d.columns:
        d[group_col] = d.index.astype(str)
    if stratify_key not in d.columns:
        d[stratify_key] = d["label"].astype(int)

    # one row per group
    g = (
        d.groupby(group_col, as_index=False)
        .agg(
            n=("label", "size"),
            label=("label", "max"),
            strat=(stratify_key, "first"),
        )
        .copy()
    )
    g = g.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    # Stratified splitting on groups (needs sklearn; fallback to simple shuffle)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        strat = g["strat"].astype(str).to_numpy()
        idx = np.arange(len(g))

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(r_va + r_te), random_state=int(seed))
        idx_tr, idx_tmp = next(sss1.split(idx, strat))
        g_tr = g.iloc[idx_tr].copy()
        g_tmp = g.iloc[idx_tmp].copy()

        # Split tmp into val/test
        if len(g_tmp) == 0:
            g_va = g_tmp
            g_te = g_tmp
        else:
            # proportion within tmp
            te_share = r_te / max(1e-9, (r_va + r_te))
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=te_share, random_state=int(seed + 1))
            idx_va, idx_te = next(sss2.split(np.arange(len(g_tmp)), g_tmp["strat"].astype(str).to_numpy()))
            g_va = g_tmp.iloc[idx_va].copy()
            g_te = g_tmp.iloc[idx_te].copy()
    except Exception:
        # simple fallback: shuffle groups then cut
        n = len(g)
        n_tr = int(round(n * r_tr))
        n_va = int(round(n * r_va))
        g_tr = g.iloc[:n_tr]
        g_va = g.iloc[n_tr : n_tr + n_va]
        g_te = g.iloc[n_tr + n_va :]

    split_map: dict[str, str] = {}
    for gg in g_tr[group_col].astype(str).tolist():
        split_map[gg] = "train"
    for gg in g_va[group_col].astype(str).tolist():
        split_map[gg] = "val"
    for gg in g_te[group_col].astype(str).tolist():
        split_map[gg] = "test"

    return d[group_col].astype(str).map(split_map).fillna("train")


def _count_no_fire(df: pd.DataFrame, split_name: str) -> int:
    sp = str(split_name).strip().lower()
    m = df["split"].astype(str).str.strip().str.lower().eq(sp)
    return int(((m) & (pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int) == 0)).sum())


def _boost_test_no_fire_by_moving_whole_groups(
    df: pd.DataFrame,
    *,
    minimum_test_no_fire: int = 100,
    locked_sources: frozenset[str] = _SPLIT_LOCKED_SOURCES,
    pin_train_eval_sources: frozenset[str] = frozenset(),
    source_hints: tuple[str, ...] = ("flame_video_nofire", "flame3", "binary"),
) -> pd.DataFrame:
    """Move whole homogeneous **no_fire** ``split_group`` buckets from train/val → test.

    Never touches ``locked_sources`` (path-authoritative rows). Skips groups that are not
    purely label 0. ``pin_train_eval_sources`` lists sources whose rows must stay on their
    current split **train** assignments (typically ``cart_aux`` when ``--cart_in_eval none``).
    """
    if df.empty or minimum_test_no_fire <= 0:
        return df
    out = df.copy()
    out["split"] = out["split"].astype(str).str.strip().str.lower().replace("", "train")
    nf = _count_no_fire(out, "test")
    if nf >= int(minimum_test_no_fire):
        return out

    uniq_sources = sorted({str(s) for s in df["source"].astype(str).tolist() if str(s)})
    tail = [s for s in uniq_sources if s not in source_hints]
    prioritized = tuple(dict.fromkeys(list(source_hints) + tail))

    def _prior_index(src: str) -> int:
        try:
            return int(prioritized.index(str(src)))
        except ValueError:
            return 999

    def _collect_candidates(dfx: pd.DataFrame, donor: str) -> list[tuple[str, int, str, str]]:
        items: list[tuple[str, int, str, str]] = []
        don = str(donor).strip().lower()
        for sg, grp in dfx.groupby(dfx["split_group"].astype(str), sort=False):
            if len(grp) == 0:
                continue
            labels = pd.to_numeric(grp["label"], errors="coerce").fillna(0).astype(int)
            if not bool((labels == 0).all()):
                continue
            src = str(grp["source"].astype(str).iloc[0])
            if src in locked_sources or src in pin_train_eval_sources:
                continue
            sp = str(grp["split"].astype(str).iloc[0]).strip().lower()
            if sp != don:
                continue
            items.append((str(sg), int(len(grp)), src, sp))
        return items

    def _sort_key(item: tuple[str, int, str, str]) -> tuple[int, int, str]:
        _sg, n, src, _sp = item
        return (_prior_index(src), -int(n), str(_sg))

    moved_groups = 0
    moved_rows = 0
    for donor in ("train", "val"):
        while nf < int(minimum_test_no_fire):
            pool = _collect_candidates(out, donor)
            if not pool:
                break
            pool.sort(key=_sort_key)
            sg, n, src, _sp = pool[0]
            out.loc[out["split_group"].astype(str) == sg, "split"] = "test"
            moved_groups += 1
            moved_rows += int(n)
            nf += int(n)
            print(
                f"[index][BOOST] moved split_group={sg!r} ({n} no_fire rows, source={src}) "
                f"{donor} → test (running test_no_fire≈{nf})",
                flush=True,
            )
        if nf >= int(minimum_test_no_fire):
            break

    if nf < int(minimum_test_no_fire) and moved_groups == 0:
        print(
            f"[index][BOOST][WARN] test no_fire={nf} < {minimum_test_no_fire} but no movable "
            f"homogeneous no_fire groups (locked={sorted(locked_sources)} "
            f"pinned={sorted(pin_train_eval_sources)}).",
            flush=True,
        )
    print(
        f"[index][BOOST] summary moved_groups={moved_groups} rows={moved_rows} "
        f"final_test_no_fire={_count_no_fire(out,'test')} target={minimum_test_no_fire}",
        flush=True,
    )
    return out


def _files_by_stem(root: Path) -> dict[str, Path]:
    """Map ``normalized_stem -> Path`` for image files under ``root``.

    Stems are normalized via :func:`_normalize_stem` (lower-case + one trailing
    modality suffix stripped) so RGB↔Thermal pairing tolerates conventions
    like ``frame_001_rgb.jpg`` ↔ ``frame_001_thermal.tif``. Collisions
    (multiple files mapping to the same key) keep the first hit.
    """
    out, _ = _files_by_stem_with_stats(root)
    return out


def _files_by_stem_with_stats(root: Path) -> tuple[dict[str, Path], dict[str, int]]:
    """Like :func:`_files_by_stem` but also reports collision / total counts."""
    out: dict[str, Path] = {}
    collisions = 0
    total = 0
    if not root.exists():
        return out, {"collisions": 0, "total": 0}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            total += 1
            key = _normalize_stem(p)
            if key in out:
                collisions += 1
                continue
            out[key] = p
    return out, {"collisions": collisions, "total": total}


def _resolve_rgb_thermal_under_class(cls_root: Path) -> tuple[Path | None, Path | None]:
    """
    Find RGB and thermal roots under one class folder, e.g.:
 - `Fire/RGB` + `Fire/Thermal`
    - `Fire/RGB/Corrected FOV` + `Fire/Thermal/Celsius TIFF` (FLAME-like)
    - `fire/rgb` + `fire/thermal` (lowercase)
    """
    rgb_names = ("RGB", "rgb", "Visible", "visible")
    th_names = ("Thermal", "thermal", "LWIR", "lwir", "IR", "ir")

    rgb_path: Path | None = None
    th_path: Path | None = None
    for n in rgb_names:
        p = cls_root / n
        if p.is_dir():
            rgb_path = p
            break
    for n in th_names:
        p = cls_root / n
        if p.is_dir():
            th_path = p
            break

    # Nested FLAME-style under RGB/
    if rgb_path is not None and rgb_path.name.lower() == "rgb":
        nested = rgb_path / "Corrected FOV"
        if nested.is_dir():
            rgb_path = nested
    # Nested under Thermal/
    if th_path is not None and th_path.name.lower() == "thermal":
        for sub in sorted(th_path.iterdir()):
            if sub.is_dir() and "celsius" in sub.name.lower():
                th_path = sub
                break

    return rgb_path, th_path


def _scan_pairs_two_trees(
    rgb_root: Path,
    th_root: Path,
    label: int,
    source: str,
    split_prefix: str,
    chunk_size: int | None = None,
    chunk_gap: int = 0,
) -> list[dict]:
    """
    Pair RGB and Thermal images by filename stem across two directory trees.

    NOTE on grouping (anti-leakage):
        - Default behaviour: ``split_group`` is set to ``split_prefix``
          (NOT per-frame). Callers MUST embed the video / scene id in
          ``split_prefix`` so that all frames belonging to the same continuous
          capture share a single group and therefore stay in the same
          train/val/test split.
        - When ``chunk_size`` is provided, frames are sorted by stem and
          bucketed into contiguous chunks of at most ``chunk_size`` rows; each
          chunk becomes its own ``split_group`` (e.g.
          ``f"{split_prefix}_chunk003"``). This is intended for sources that
          are a single long video — chunking lets all three splits receive a
          share of the source while preserving temporal locality. The first
          ``chunk_gap`` frames at the start of every chunk are dropped to
          insert a buffer between adjacent chunks (so neighbouring frames
          across a chunk boundary do not leak content into different splits).
        - ``key`` is dataset-prefixed and unique per row so it is safe to use
          as an identifier across the whole index.
    """
    rgb_map = _files_by_stem(rgb_root)
    th_map = _files_by_stem(th_root)
    rows: list[dict] = []
    if not rgb_map or not th_map:
        return rows

    use_chunks = bool(chunk_size and int(chunk_size) > 0)
    sorted_keys = sorted(rgb_map.keys()) if use_chunks else list(rgb_map.keys())

    for i, k in enumerate(sorted_keys):
        rgb_path = rgb_map[k]
        th_path = th_map.get(k)
        if th_path is None:
            continue
        if use_chunks:
            cs = int(chunk_size)
            pos_in_chunk = i % cs
            if chunk_gap > 0 and pos_in_chunk < int(chunk_gap):
                # buffer zone at the start of each chunk; drop frame to keep
                # a temporal gap between adjacent chunks
                continue
            chunk_id = i // cs
            sg = f"{split_prefix}_chunk{chunk_id:03d}"
            key_full = f"{split_prefix}_chunk{chunk_id:03d}_{k}"
        else:
            sg = str(split_prefix)
            key_full = f"{split_prefix}_{k}"
        rows.append(
            {
                "path_rgb": str(rgb_path),
                "path_th": str(th_path),
                "label": int(label),
                "label_fire": int(label),
                "source": str(source),
                "key": key_full,
                "split_group": sg,
            }
        )
    return rows


def _scan_pairs_flame_root(flame_root: Path) -> list[dict]:
    """
    Scanner tailored for this repo's `data/flame3/` layout.

    Supported sources under `flame_root`:
    - `Fire/RGB/Corrected FOV/*` + `Fire/Thermal/Celsius TIFF/*`
    - `No Fire/RGB/Corrected FOV/*` + `No Fire/Thermal/Celsius TIFF/*`

    Standalone multimodal BINARY (``BINARY_ROOT/train|val|test``) is indexed
    separately. ``Dataset/`` here is RGB-only and is intentionally omitted.
    """
    rows: list[dict] = []
    if not flame_root.exists():
        return rows

    # FLAME main — flame3 is a single continuous video per class on disk
    # (filenames are pure sequential ids). We split it across train/val/test
    # at the block level: ``chunk_size`` consecutive frames form one group and
    # ``chunk_gap`` frames at the start of every chunk are dropped to keep a
    # temporal buffer between adjacent chunks. This is the only way to get
    # both fire and no_fire flame3 examples into all three splits without
    # frame-level leakage.
    rows += _scan_pairs_two_trees(
        rgb_root=flame_root / "Fire" / "RGB" / "Corrected FOV",
        th_root=flame_root / "Fire" / "Thermal" / "Celsius TIFF",
        label=1,
        source="flame3",
        split_prefix="flame3_fire",
        chunk_size=20,
        chunk_gap=2,
    )
    rows += _scan_pairs_two_trees(
        rgb_root=flame_root / "No Fire" / "RGB" / "Corrected FOV",
        th_root=flame_root / "No Fire" / "Thermal" / "Celsius TIFF",
        label=0,
        source="flame3",
        split_prefix="flame3_nofire",
        chunk_size=20,
        chunk_gap=2,
    )

    return rows


def _binary_class_directories(binary_parent: Path) -> list[tuple[str, int]]:
    """Detect ``fire`` / ``no_fire`` class folders immediately under ``binary_parent``."""

    fire_aliases = {"fire"}
    # Kaggle/binary bundle: Fire + No_Fire (normalized), no-fire, NF, etc.
    nofire_aliases = {"no_fire", "no fire", "nofire", "no-fire", "nf", "not_fire", "not fire"}
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    try:
        children = [p for p in binary_parent.iterdir() if p.is_dir()]
    except OSError:
        return out
    for p in children:
        n = p.name.strip().lower().replace(" ", "_").replace("-", "_")
        if n in fire_aliases:
            lab = 1
        elif n in nofire_aliases:
            lab = 0
        else:
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append((p.name, lab))
    return out


def _pair_rgb_th_dirs_under_parent(
    parent: Path,
    *,
    label: int,
    source: str,
    split_prefix: str,
) -> list[dict]:
    """Pair modality folders directly under ``parent`` (*_rgb + *_thermal or FLAME nested layout)."""

    rr, tt = _resolve_rgb_thermal_under_class(parent)
    if rr is not None and tt is not None:
        return _scan_pairs_two_trees(
            rgb_root=rr,
            th_root=tt,
            label=label,
            source=source,
            split_prefix=split_prefix,
        )

    subdirs = [p for p in parent.iterdir() if p.is_dir()]

    def _is_rgb_dir(d: Path) -> bool:
        n = d.name.lower()
        return n.endswith("_rgb") or n.endswith("_w") or "_rgb" in n

    def _is_th_dir(d: Path) -> bool:
        n = d.name.lower()
        return (
            n.endswith("_thermal")
            or n.endswith("_thermal16")
            or n.endswith("_t")
            or "_thermal" in n
        )

    def _base_key(d: Path) -> str:
        n = d.name
        for s in ("_rgb", "_thermal16", "_thermal", "_w", "_t"):
            n = n.replace(s, "")
        return n

    rows: list[dict] = []
    rgb_by_base = {_base_key(d): d for d in subdirs if _is_rgb_dir(d)}
    th_by_base = {_base_key(d): d for d in subdirs if _is_th_dir(d)}
    for base, rdir in rgb_by_base.items():
        tdir = th_by_base.get(base)
        if tdir is None:
            continue
        rows += _scan_pairs_two_trees(
            rgb_root=rdir,
            th_root=tdir,
            label=label,
            source=source,
            split_prefix=f"{split_prefix}_{base}",
        )
    return rows


def _scan_binary_split_subtree(binary_split_folder: Path, preset_split: str) -> list[dict]:
    """
    Scan one preset split subtree (``BINARY_ROOT/train`` …) with ``Fire``/``No_Fire`` layouts.

    Supports:

    - ``Fire/*_rgb`` + ``Fire/*_thermal`` siblings (flat),

    - ``Fire/<session_id>/{*_rgb,*_thermal}`` (nested under one folder per drone video).
    """

    rows_out: list[dict] = []
    if not binary_split_folder.is_dir():
        return rows_out

    sp = str(preset_split).strip().lower()

    for cls_name, lab in _binary_class_directories(binary_split_folder):
        cls_root = binary_split_folder / cls_name

        pk = "fire" if lab == 1 else "nofire"
        base_px = f"binaryroot_{sp}_{pk}"

        # Flat FLAME-like or flat *_rgb / *_thermal under Fire | No_Fire.
        chunk = _pair_rgb_th_dirs_under_parent(
            cls_root,
            label=lab,
            source="binary_root",
            split_prefix=base_px,
        )
        if chunk:
            for r in chunk:
                r["split"] = sp
            rows_out.extend(chunk)

        # Nested: e.g. …/Fire/<drone_session>/*_rgb next to *_thermal.
        try:
            nested_roots = [ch for ch in cls_root.iterdir() if ch.is_dir()]
        except OSError:
            nested_roots = []
        for nx in nested_roots:
            sub = _pair_rgb_th_dirs_under_parent(
                nx,
                label=lab,
                source="binary_root",
                split_prefix=f"{base_px}_{nx.name}",
            )
            if not sub:
                continue
            for r in sub:
                r["split"] = sp
            rows_out.extend(sub)

    return rows_out


def _scan_binary_dataset_root_preset(binary_dataset_root: Path) -> tuple[list[dict], dict[str, object]]:
    """
    Expect ``BINARY_ROOT/train|val|test`` subtrees containing fire/no_fire RGB+Thermal pairs.

    Rows include an authoritative ``split`` column aligned with on-disk preset folders.
    """
    if not binary_dataset_root.is_dir():
        return [], {"skipped": True, "reason": "not_dir"}

    presets_ok = [(binary_dataset_root / p).is_dir() for p in ("train", "val", "test")]
    if not any(presets_ok):
        print(
            f"[binary_root] skipped: no train/, val/, or test/ folders under "
            f"{binary_dataset_root.resolve()} (expected preset split layout).",
            flush=True,
        )
        return [], {"skipped": True, "reason": "no_preset_split_folders"}

    all_rows: list[dict] = []
    per_split_counts: dict[str, int] = {}
    label_per_split_by_s: defaultdict[str, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    for p in ("train", "val", "test"):
        sub = binary_dataset_root / p
        if not sub.is_dir():
            per_split_counts[p] = 0
            continue
        got = _scan_binary_split_subtree(sub, p)
        per_split_counts[p] = len(got)
        for row in got:
            label_per_split_by_s[p][int(row["label"])] += 1
        all_rows.extend(got)

    return all_rows, {
        "total": len(all_rows),
        "per_split": per_split_counts,
        "label_by_split": {k: dict(v) for k, v in label_per_split_by_s.items()},
    }


def _scan_flame_video_frames(root: Path) -> tuple[list[dict], int]:
    """
    Include extracted RGB+thermal frame pairs as real no_fire negatives.

    Layout:
      data/flame_video_frames/rgb/*.jpg
      data/flame_video_frames/thermal/*.jpg
    Pairing: filename stem match.
    """
    rgb_dir = root / "rgb"
    th_dir = root / "thermal"
    if not (rgb_dir.is_dir() and th_dir.is_dir()):
        return [], 0
    rgb_map = _files_by_stem(rgb_dir)
    th_map = _files_by_stem(th_dir)
    missing = 0
    rows: list[dict] = []
    for k, rgb_p in rgb_map.items():
        th_p = th_map.get(k)
        if th_p is None:
            missing += 1
            continue
        rows.append(
            {
                "path_rgb": str(rgb_p),
                "path_th": str(th_p),
                "label": 0,
                "label_fire": 0,
                "source": "flame_video_nofire",
                "key": f"flame_video_nofire_{k}",
                # Per-frame placeholder; the pair-level grouping is set later
                # in ``build_master_index`` (B-branch) so all frames from the
                # same recording pair share one group and split.
                "split_group": f"flame_video_nofire_{k}",
            }
        )
    if missing:
        print(f"[flame_video_nofire] missing_pairs={missing} (skipped)", flush=True)
    return rows, missing


def _cart_group_from_path(rgb_path: Path, color_root: Path) -> str:
    """
    Best-effort 'flight/session' grouping for CART so a capture doesn't leak across splits.

    Rules:
      - Prefer a meaningful prefix from filename stem (token before '_' / '-')
      - Otherwise fall back to (dataset + parent folder) which is stable
    """
    try:
        rel = rgb_path.relative_to(color_root)
    except Exception:
        rel = Path(rgb_path.name)

    # If CART has nested folders, use the top-level folder as a stable group.
    parts = list(rel.parts)
    if len(parts) >= 2:
        grp = str(parts[0]).strip().lower()
        if grp:
            return grp

    stem = rgb_path.stem.strip().lower()
    for sep in ("_", "-"):
        if sep in stem:
            tok = stem.split(sep, 1)[0].strip().lower()
            if tok and not tok.isdigit():
                return tok

    parent = rgb_path.parent.name.strip().lower()
    # Fallback rule: use parent folder (caller will prefix dataset name).
    return parent if parent else "root"


def _scan_cart_pairs(cart_root: Path, max_cart_samples: int = 500) -> list[dict]:
    """
    CART auxiliary negatives:
      data/cart/color/      -> RGB
      data/cart/thermal16/  -> thermal (16-bit preferred)

    Excludes:
      - thermal8/
      - thermal_ann_overlay/
      - annotations/
    """
    cart_root = Path(cart_root)
    color_root = cart_root / "color"
    th16_root = cart_root / "thermal16"
    if not (color_root.is_dir() and th16_root.is_dir()):
        return []

    rgb_map, rgb_stats = _files_cart_side(color_root)
    th_map, th_stats = _files_cart_side(th16_root)
    if not rgb_map or not th_map:
        print(
            f"[cart] empty side: rgb_total={rgb_stats['total']} th_total={th_stats['total']}",
            flush=True,
        )
        return []

    # Pairing diagnostics (rgb_only, th_only, matched, collisions, pair_rate).
    rgb_only = sum(1 for k in rgb_map if k not in th_map)
    th_only = sum(1 for k in th_map if k not in rgb_map)
    matched = sum(1 for k in rgb_map if k in th_map)
    union = max(1, len(set(rgb_map) | set(th_map)))
    pair_rate = float(matched) / float(union)
    print(
        f"[cart] pairing: matched={matched} rgb_only={rgb_only} th_only={th_only} "
        f"collisions(rgb={rgb_stats['collisions']}, th={th_stats['collisions']}) "
        f"rate={pair_rate:.3f}",
        flush=True,
    )
    if matched == 0:
        print("[cart][WARN] pairing failed: no matching RGB/thermal16 pairs found", flush=True)
    if pair_rate < 0.90:
        print(
            f"[cart][WARN] pairing rate {pair_rate:.3f} < 0.90 — many files dropped; "
            f"check stem naming conventions in {color_root} vs {th16_root}.",
            flush=True,
        )

    paired: list[tuple[str, Path, Path]] = []
    for stem, rgb_p in rgb_map.items():
        th_p = th_map.get(stem)
        if th_p is None:
            continue
        paired.append((stem, rgb_p, th_p))

    # Deterministic: sort by (group, stem)
    paired.sort(key=lambda x: (_cart_group_from_path(x[1], color_root), x[0]))

    # Deterministic downsample to avoid flooding no_fire pool.
    max_n = int(max_cart_samples) if max_cart_samples is not None else 500
    max_n = 0 if max_n < 0 else max_n
    if max_n and len(paired) > max_n:
        # Uniform selection across the sorted list; no random shuffle.
        idx = np.linspace(0, len(paired) - 1, num=max_n, dtype=int).tolist()
        # ensure strictly increasing unique indices
        keep = []
        last = -1
        for i in idx:
            ii = int(i)
            if ii <= last:
                continue
            keep.append(ii)
            last = ii
        paired = [paired[i] for i in keep]

    rows: list[dict] = []
    for stem, rgb_p, th_p in paired:
        grp = _cart_group_from_path(rgb_p, color_root)
        # key must be globally unique and deterministic
        key = f"cart_{grp}_{stem}"
        rows.append(
            {
                "path_rgb": str(rgb_p),
                "path_th": str(th_p),
                "label": 0,
                "label_fire": 0,
                "source": "cart_aux",
                "key": key,
                "split_group": f"cart_{grp}",
            }
        )
    print(f"[cart] added {len(rows)} paired RGB-thermal16 no_fire samples", flush=True)
    return rows


def _load_binary_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # normalize expected column names
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("rgb_path", "path_rgb", "rgb"):
            colmap[c] = "path_rgb"
        if cl in ("thermal_path", "path_th", "path_thermal", "thermal"):
            colmap[c] = "path_th"
        if cl in ("label", "y", "target", "fire"):
            colmap[c] = "label"
    if colmap:
        df = df.rename(columns=colmap)
    if "path_rgb" not in df.columns or "path_th" not in df.columns or "label" not in df.columns:
        return None
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["label_fire"] = df["label"].astype(int)
    df["source"] = "binary"
    # Use the parent-directory (typically the video/session id) as the group so
    # frames from the same recording cannot be split across train/val/test.
    def _video_id_from_path(p: str) -> str:
        try:
            parent = Path(str(p)).parent.name
        except Exception:
            parent = ""
        parent = parent.lower()
        for s in ("_rgb", "_thermal", "_w", "_t"):
            if parent.endswith(s):
                parent = parent[: -len(s)]
                break
        return parent or "binary"

    stems = df["path_rgb"].astype(str).map(lambda x: Path(x).stem)
    videos = df["path_rgb"].astype(str).map(_video_id_from_path)
    df["key"] = "binary_" + videos.astype(str) + "_" + stems.astype(str)
    df["split_group"] = "binary_" + videos.astype(str)
    return df[["path_rgb", "path_th", "label", "label_fire", "source", "key", "split_group"]].copy()


def _video_base_from_path(value: object) -> str:
    """Heuristic ``<full_parent_path>/<stem-without-trailing-digits>`` used to
    detect sequence-level leakage where two frames from the same recording end
    up in different splits but have different file names. Using the full parent
    path (not just ``parent.name``) avoids false positives across class folders
    that happen to share a leaf directory name (e.g. ``Fire/RGB/Corrected FOV``
    vs ``No Fire/RGB/Corrected FOV``)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    p = Path(str(value).replace("\\", "/"))
    stem = p.stem.lower()
    stripped = _FRAME_TAIL_RE.sub("", stem)
    parent_path = p.parent.as_posix().lower()
    return f"{parent_path}/{stripped}"


def _assert_no_split_leakage(df: pd.DataFrame) -> None:
    """Hard guard: refuses to let ``build_master_index`` write the parquet if
    any of the leakage criteria would be flagged by ``scripts/check_leakage.py``.

    Raises ``SystemExit(1)`` with a detailed message on failure.
    """
    required = {"path_rgb", "path_th", "key", "split_group", "split", "source"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[leakage-guard] index is missing columns: {missing}")

    def _norm(p: object) -> str:
        return str(p).replace("\\", "/").lower() if p is not None else ""

    # ``is_strict`` checks must hold for every row. The two ``video_stem_*``
    # checks are heuristic (parent_path + stripped numeric suffix) and are
    # too coarse for sources we deliberately split at the block level — see
    # ``_BLOCK_SPLIT_OPT_IN_SOURCES``. Rows from those sources are dropped
    # from the heuristic checks but still validated by the strict ones
    # (path_rgb / path_th / key / split_group).
    checks: list[tuple[str, pd.Series, bool]] = [
        ("path_rgb", df["path_rgb"].map(_norm), True),
        ("path_th", df["path_th"].map(_norm), True),
        ("key", df["key"].astype(str), True),
        ("split_group", df["split_group"].astype(str), True),
        ("video_stem_rgb", df["path_rgb"].map(_video_base_from_path), False),
        ("video_stem_th", df["path_th"].map(_video_base_from_path), False),
    ]

    leaked: list[str] = []
    for name, series, is_strict in checks:
        scoped = df.assign(_k=series)
        if not is_strict and _BLOCK_SPLIT_OPT_IN_SOURCES:
            scoped = scoped[~scoped["source"].astype(str).isin(_BLOCK_SPLIT_OPT_IN_SOURCES)]
        nun = scoped.groupby("_k")["split"].nunique()
        bad = nun[nun > 1]
        if len(bad) == 0:
            print(f"[leakage-guard] OK  {name:<16s} (no cross-split overlap)")
            continue
        sub = scoped[scoped["_k"].isin(bad.index)]
        n_groups = int(bad.shape[0])
        n_rows = int(len(sub))
        leaked.append(f"{name}: {n_groups} groups / {n_rows} rows")
        print(
            f"[leakage-guard] FAIL {name:<16s} groups={n_groups} rows={n_rows}"
        )
        per_src = sub.groupby("source")["_k"].nunique().sort_values(ascending=False)
        for src, cnt in per_src.items():
            print(f"    - {src:<20s} {cnt}")
        print("    examples:")
        cols = ["split", "source", "label", "key", "split_group", "_k"]
        with pd.option_context("display.max_colwidth", 80, "display.width", 180):
            print(sub[cols].head(10).to_string(index=False))

    if leaked:
        raise SystemExit(
            "[leakage-guard] Aborting build: temporal/identity leakage detected:\n  - "
            + "\n  - ".join(leaked)
            + "\nFix split_group / key construction before training."
        )


def _summarize_disk_paths_maybe_missing(df: pd.DataFrame) -> None:
    """Print missing-file counts (capped iteration for huge indices)."""
    n = len(df)
    cap = min(n, 200_000)
    if cap == 0:
        print("[paths] missing_rgb=0 missing_th=0 (empty index)", flush=True)
        return
    if n > cap:
        dx = df.sample(n=cap, random_state=0)
        print(f"[paths] sampling {cap:,} rows for existence check ({n:,} total)", flush=True)
    else:
        dx = df
    miss_rgb = (~dx["path_rgb"].map(lambda x: Path(str(x)).expanduser().is_file())).sum()
    miss_th = (~dx["path_th"].map(lambda x: Path(str(x)).expanduser().is_file())).sum()
    print(
        f"[paths] missing_rgb={int(miss_rgb)} missing_th={int(miss_th)} (checked_rows={cap})",
        flush=True,
    )


def build_master_index(
    out_parquet: Path | None = None,
    out_legacy_csv: Path | None = None,
    scan_custom_roots: list[Path] | None = None,
    include_vegetation_stub: bool = False,
    max_cart_samples: int = 500,
    cart_in_eval: str = "none",
    cart_root: str | Path | None = None,
    binary_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build ``master_index.parquet`` and legacy ``outputs/flame_index.csv``.

    Path precedence (standalone datasets): ``FLAME_*`` env overrides ``CLI`` overrides local defaults.

    Vegetation CLI is reserved — does not change indexing until a parser lands.
    """
    _ = include_vegetation_stub

    out_parquet = Path(out_parquet or MASTER_INDEX_PARQUET)
    out_legacy_csv = Path(out_legacy_csv or FLAME_INDEX_CSV)

    flame_p = Path(FLAME_ROOT).expanduser().resolve()
    if not flame_p.exists():
        print(f"[flame3] skipped: root not found ({flame_p})", flush=True)
    else:
        print(f"[flame3] root={flame_p}", flush=True)

    roots = [flame_p]
    if scan_custom_roots:
        roots.extend(Path(r).expanduser().resolve() for r in scan_custom_roots)
        print(f"[flame3] extra_scan_roots={[str(r) for r in roots[1:]]}", flush=True)

    all_rows: list[dict] = []
    for root in roots:
        all_rows.extend(_scan_pairs_flame_root(root))

    df = pd.DataFrame(all_rows)

    resolved_bin_root, bin_from = _resolve_root_env_cli_default("FLAME_BINARY_ROOT", binary_root, Path(BINARY_ROOT))
    resolved_bin_exp = resolved_bin_root.expanduser().resolve()
    binary_stats: dict[str, object] = {}
    if not resolved_bin_exp.is_dir():
        print(f"[binary_root] skipped: root not found ({resolved_bin_exp}) [{bin_from}]", flush=True)
    else:
        print(f"[binary_root] root={resolved_bin_exp} [{bin_from}]", flush=True)
        bin_rows_add, binary_stats = _scan_binary_dataset_root_preset(resolved_bin_exp)
        if isinstance(binary_stats.get("skipped"), bool) and binary_stats["skipped"]:
            pass  # preset missing — already warned
        else:
            lbl = binary_stats.get("label_by_split", {})
            pst = binary_stats.get("per_split", {})
            print(
                f"[binary_root] preset split summary rows={binary_stats.get('total', len(bin_rows_add))} "
                f"per_split={pst} labels_by_split={lbl}",
                flush=True,
            )
        if bin_rows_add:
            df = pd.concat([df, pd.DataFrame(bin_rows_add)], ignore_index=True)

    vid_fp = Path(FLAME_VIDEO_FRAMES_ROOT).expanduser().resolve()
    if vid_fp.exists():
        print(f"[flame_video_nofire] root={vid_fp}", flush=True)
        vid_rows, _vid_missing = _scan_flame_video_frames(vid_fp)
    else:
        print(f"[flame_video_nofire] skipped: root not found ({vid_fp})", flush=True)
        vid_rows = []

    if vid_rows:
        df = pd.concat([df, pd.DataFrame(vid_rows)], ignore_index=True)

    cart_default = Path(DATA_ROOT) / "cart"
    cart_hint, cart_from = _resolve_root_env_cli_default("FLAME_CART_ROOT", cart_root, cart_default)
    cart_chk = cart_hint.expanduser().resolve()
    cart_rows: list[dict] = []
    if not cart_chk.exists():
        print(f"[cart] skipped: root not found ({cart_chk}) [{cart_from}]", flush=True)
    else:
        cart_base = _resolve_cart_layout_base(cart_chk)
        if cart_base is None:
            alt_try = cart_chk / "cart"
            print(
                f"[cart] skipped: expected color/+thermal16 layout under {cart_chk}; "
                f"also tried {alt_try}",
                flush=True,
            )
        else:
            print(f"[cart] resolved root={cart_base} [{cart_from}]", flush=True)
            cart_rows = _scan_cart_pairs(cart_base, max_cart_samples=max_cart_samples)

    if cart_rows:
        df = pd.concat([df, pd.DataFrame(cart_rows)], ignore_index=True)
        print(
            "[index] hint: CART(train-only negatives) stacks with weighted sampling — "
            "prefer no_fire_weight in [1.0, 1.2] vs 2.0 when cart_aux rows are enabled.",
            flush=True,
        )

    emb_csv_path = Path(FLAME_EMBEDDED_CSV).expanduser().resolve()
    if emb_csv_path.exists():
        print(f"[flame_binary_csv] root={emb_csv_path}", flush=True)
        bin_df = _load_binary_csv(emb_csv_path)
    else:
        print(f"[flame_binary_csv] skipped: file not found ({emb_csv_path})", flush=True)
        bin_df = None
    if bin_df is not None and not bin_df.empty:
        df = pd.concat([df, bin_df], ignore_index=True)

    if df.empty:
        raise SystemExit(
            "Index oluşturulamadı: hiçbir (rgb,thermal,label) eşleşmesi bulunamadı. "
            "Klasör yapıları (flame3, binary/train|val|test, ...) kontrol edin."
        )

    # ----------------------------
    # Split design (dataset-aware)
    # ----------------------------
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["label_fire"] = df["label"].astype(int)
    if "split_group" not in df.columns:
        df["split_group"] = df["source"].astype(str) + "_" + df["key"].astype(str)

    if "split" in df.columns:
        df["split"] = (
            df["split"].fillna("").astype(str).replace({"nan": "", "NaT": "", "None": ""}).fillna("")
        )
    else:
        df["split"] = ""

    # A) flame3 optional path-encoded split (normally empty stem paths don't match)
    for src in ["flame3"]:
        m = df["source"].astype(str) == src
        if not int(m.sum()):
            continue
        inferred = df.loc[m, "path_rgb"].map(_infer_official_split_from_path)
        inferred2 = df.loc[m, "path_th"].map(_infer_official_split_from_path) if "path_th" in df.columns else None
        if inferred2 is not None:
            inferred = inferred.fillna(inferred2)
        df.loc[m, "split"] = inferred.fillna("").astype(str)

    # B) flame_video_nofire: pair-level split (bias test toward ~target_test_frames no_fire rows)
    mv = df["source"].astype(str) == "flame_video_nofire"
    if int(mv.sum()):
        pair_series = df.loc[mv, "path_rgb"].astype(str).map(_extract_video_pair_id)
        pair_counts = pair_series.value_counts().astype(int).to_dict()
        pairs_sorted = sorted(pair_counts.keys())

        p_train, p_val, p_test = _partition_flame_video_pairs(
            pairs_sorted,
            {k: int(pair_counts[k]) for k in pair_counts},
            target_test_frames=420,
            max_test_frame_frac=0.52,
        )

        fr_train = int(sum(pair_counts[p] for p in p_train if p in pair_counts))
        fr_val = int(sum(pair_counts[p] for p in p_val if p in pair_counts))
        fr_test = int(sum(pair_counts[p] for p in p_test if p in pair_counts))
        print(
            f"[flame_video_nofire] pair split clips: "
            f"train={len(p_train)} val={len(p_val)} test={len(p_test)} "
            f"(frames train={fr_train} val={fr_val} test={fr_test})",
            flush=True,
        )
        if len(p_train) == 0:
            print(
                "[flame_video_nofire][WARN] zero pairs assigned to TRAIN — "
                "model will not learn this no_fire domain. Check pair count "
                "and _partition_flame_video_pairs policy.",
                flush=True,
            )

        pid_to_side: dict[str, str] = {}
        for p in p_train:
            pid_to_side[p] = "train"
        for p in p_val:
            pid_to_side[p] = "val"
        for p in p_test:
            pid_to_side[p] = "test"

        sid = pair_series.map(pid_to_side)
        df.loc[mv, "split"] = sid.values
        df.loc[mv, "split_group"] = "flame_video_nofire_pair_" + pair_series.astype(str)

    # D) Rows without authoritative split (binary preset / cart train-only applied later cover most cases)
    has_split_mask = df["split"].astype(str).str.strip().isin(["train", "val", "test"])
    remain_mask = ~has_split_mask
    if int(remain_mask.sum()):
        sub = df.loc[remain_mask].copy()
        sub["strat_key"] = sub["label"].astype(str) + "_" + sub["source"].astype(str)
        spl_series = _stratified_group_split(
            sub,
            ratios=(0.70, 0.15, 0.15),
            seed=42,
            stratify_key="strat_key",
            group_col="split_group",
        )
        df.loc[remain_mask, "split"] = spl_series
        df.drop(columns=["strat_key"], inplace=True, errors="ignore")

    def _has_both_labels(dfx: pd.DataFrame) -> bool:
        g = dfx.groupby(["split", "label"]).size().unstack(fill_value=0)
        for sp in ["train", "val", "test"]:
            if sp not in g.index:
                return False
            if 0 not in g.columns or 1 not in g.columns:
                return False
            if int(g.loc[sp, 0]) == 0 or int(g.loc[sp, 1]) == 0:
                return False
        return True

    if not _has_both_labels(df):
        movable_mask = ~df["source"].astype(str).isin(_SPLIT_LOCKED_SOURCES).to_numpy()
        if movable_mask.any():
            m_sub = df.loc[movable_mask].copy()
            m_sub["strat_key"] = m_sub["label"].astype(str) + "_" + m_sub["source"].astype(str)
            spl2 = _stratified_group_split(
                m_sub,
                ratios=(0.70, 0.15, 0.15),
                seed=123,
                stratify_key="strat_key",
                group_col="split_group",
            )
            df.loc[movable_mask, "split"] = spl2
            df.drop(columns=["strat_key"], inplace=True, errors="ignore")
            if not _has_both_labels(df):
                print(
                    "[index][WARN] stratified splits still imbalance after remap — "
                    "check source availability per label.",
                    flush=True,
                )
        else:
            print(
                "[index][WARN] splits lack both labels but locked sources prevent global rebalance.",
                flush=True,
            )

    # CART eval policy AFTER splits (override stratified assigns if needed)
    cart_mask = df["source"].astype(str) == "cart_aux"
    n_cart = int(cart_mask.sum())
    policy = (cart_in_eval or "none").strip().lower()
    if n_cart > 0 and policy != "none":
        print(
            "[index][WARN] cart_aux in val/test artificially inflates eval metrics — "
            "recommended: --cart_in_eval none (train-only CART).",
            flush=True,
        )
    if n_cart > 0:
        if policy not in {"none", "val", "test", "both"}:
            print(f"[cart][WARN] unknown cart_in_eval={cart_in_eval!r}; falling back to 'none'.", flush=True)
            policy = "none"
        if policy == "none":
            df.loc[cart_mask, "split"] = "train"
            print(
                "[cart_aux] train-only (--cart_in_eval none): keeping all cart_aux rows on split=train; "
                "test_no_fire boosts will skip cart_aux split_groups.",
                flush=True,
            )
        elif policy in ("val", "test"):
            df.loc[cart_mask, "split"] = policy
        print(f"[cart] policy={policy} | rows={n_cart} (split applied)", flush=True)

    pin_cart_boost = frozenset({"cart_aux"}) if (n_cart > 0 and policy == "none") else frozenset()
    df = _boost_test_no_fire_by_moving_whole_groups(
        df,
        minimum_test_no_fire=100,
        pin_train_eval_sources=pin_cart_boost,
    )

    print("=== split objectives (row counts; fire=yes label 1 / no-fire=label 0) ===")
    for sp in ("train", "val", "test"):
        m_sp = df["split"].astype(str).str.strip().str.lower().eq(sp)
        lab = pd.to_numeric(df.loc[m_sp, "label"], errors="coerce").fillna(0).astype(int)
        n_fire = int((lab == 1).sum())
        n_nf = int((lab == 0).sum())
        print(f"[split:{sp}] total={len(lab)} fire={n_fire} no_fire={n_nf}", flush=True)

    print("=== split objectives (sources: rows per split) ===")
    for sp in ("train", "val", "test"):
        m_sp = df["split"].astype(str).str.strip().str.lower().eq(sp)
        sc = df.loc[m_sp, "source"].astype(str).value_counts()
        tops = "; ".join(f"{k}:{int(sc[k])}" for k in sc.index[:12])
        more = f" (+{len(sc) - 12} more)" if len(sc) > 12 else ""
        print(f"[split:{sp}] sources | {tops}{more}", flush=True)

    print("=== split x label ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())

    print("=== split x source ===")
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0).to_string())

    print("=== split x source x label ===")
    _ssl = pd.crosstab(
        index=[df["split"].astype(str), df["source"].astype(str)],
        columns=df["label"].astype(str),
        dropna=False,
    )
    print(_ssl.to_string())

    print("=== distinct split_group counts by source ===")
    grp_stats = df.groupby("source")["split_group"].nunique().sort_values(ascending=False)
    print(grp_stats.to_string())

    _summarize_disk_paths_maybe_missing(df)

    train_no_fire = _count_no_fire(df, "train")
    val_no_fire = _count_no_fire(df, "val")
    test_no_fire = _count_no_fire(df, "test")
    print(
        f"[index] no_fire row counts train={train_no_fire} val={val_no_fire} test={test_no_fire}",
        flush=True,
    )
    if test_no_fire < 100:
        print(
            f"[index][WARN] test no_fire={test_no_fire} (<100): "
            "FPR / specificity on test will be unreliable — add no_fire clips or widen movable sources.",
            flush=True,
        )
    if val_no_fire < 200:
        print(
            f"[index][WARN] val no_fire={val_no_fire} (<200): "
            "early stopping / checkpoint selection based on val FPR may be optimistic.",
            flush=True,
        )
    for sp in ["train", "val", "test"]:
        sub = df[df["split"].astype(str) == sp]
        if len(sub) == 0:
            continue
        share = sub["source"].astype(str).value_counts(normalize=True)
        if len(share) and float(share.iloc[0]) > 0.70:
            print(
                f"[index][WARN] source '{share.index[0]}' dominates {sp} "
                f"({100.0 * float(share.iloc[0]):.1f}%) — global metrics will reflect this source disproportionately.",
                flush=True,
            )

    print("=== Leakage guard ===")
    _assert_no_split_leakage(df)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_legacy_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_legacy_csv, index=False)
    return df
