"""Data leakage audit for ``master_index.parquet``.

Checks whether the same physical sample (RGB path, thermal path, key,
split_group) or the same underlying video / scene (heuristic on filename
stems) appears in more than one of train / val / test splits.

Usage::

    python scripts/check_leakage.py
    python scripts/check_leakage.py --csv /kaggle/working/data/master_index.parquet
    FLAME_MASTER_INDEX=/.../master_index.parquet python scripts/check_leakage.py

Path resolution order:

1. ``--csv`` CLI argument (if provided)
2. ``FLAME_MASTER_INDEX`` environment variable
3. ``<project_root>/data/master_index.parquet`` (default)

Exit code is 1 when any kind of leakage is detected, 0 otherwise.
A detailed CSV is written to ``outputs/leakage_report.csv`` (or
``$FLAME_OUTPUTS_DIR/leakage_report.csv`` when that env is set).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "master_index.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "leakage_report.csv"


def _resolve_index_path(cli_csv: str | None) -> Path:
    """``--csv`` > ``FLAME_MASTER_INDEX`` env > default."""
    if cli_csv:
        return Path(cli_csv).expanduser().resolve()
    env = os.environ.get("FLAME_MASTER_INDEX", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_INDEX_PATH


def _resolve_report_path() -> Path:
    """``$FLAME_OUTPUTS_DIR/leakage_report.csv`` > default."""
    env = os.environ.get("FLAME_OUTPUTS_DIR", "").strip()
    if env:
        return (Path(env).expanduser() / "leakage_report.csv").resolve()
    return DEFAULT_REPORT_PATH

REQUIRED_COLUMNS = [
    "path_rgb",
    "path_th",
    "label",
    "source",
    "key",
    "split_group",
    "split",
]

# Trailing frame number pattern (e.g. ``_0001``, ``-00012``, ``00345``).
_FRAME_TAIL = re.compile(r"[-_]?\d+$")


def _normalize_path(value: object) -> str:
    """Normalize a path for cross-platform comparison."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().replace("\\", "/").lower()
    return s


def _base_id_from_path(value: object) -> str:
    """Heuristic base-id used to detect video / sequence leakage.

    The base id is ``<full_parent_path>/<stem_without_trailing_frame_digits>``.
    Using the *full* parent path keeps Fire/No-Fire sibling directories apart
    even when the leaf directory name (e.g. ``Corrected FOV``) is identical.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    p = Path(str(value).replace("\\", "/"))
    stem = p.stem.lower()
    stripped = _FRAME_TAIL.sub("", stem)
    parent_path = p.parent.as_posix().lower()
    return f"{parent_path}/{stripped}"


def _collisions(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Return the subset of ``df`` whose ``key_col`` value appears in
    more than one distinct ``split``."""
    counts = df.groupby(key_col)["split"].nunique()
    bad_keys = counts[counts > 1].index
    if len(bad_keys) == 0:
        return df.iloc[0:0]
    sub = df[df[key_col].isin(bad_keys)].copy()
    return sub.sort_values([key_col, "split"]).reset_index(drop=True)


def _summarize(name: str, key_col: str, collisions: pd.DataFrame) -> dict:
    n_groups = collisions[key_col].nunique() if not collisions.empty else 0
    n_rows = len(collisions)
    print()
    print("=" * 72)
    print(f"[{name}] leakage check on column: {key_col}")
    print("=" * 72)
    if collisions.empty:
        print("  OK - no overlap across splits.")
        return {
            "check": name,
            "key_col": key_col,
            "n_leaked_groups": 0,
            "n_leaked_rows": 0,
        }

    print(f"  Leaked groups : {n_groups}")
    print(f"  Leaked rows   : {n_rows}")

    by_source = (
        collisions.groupby("source")[key_col]
        .nunique()
        .sort_values(ascending=False)
    )
    print("  Per-source breakdown (distinct leaked groups):")
    for src, cnt in by_source.items():
        print(f"    - {src:<20s} {cnt}")

    sample_cols = [
        "split",
        "source",
        "label",
        "key",
        "split_group",
        key_col if key_col not in {"key", "split_group"} else "path_rgb",
    ]
    sample_cols = list(dict.fromkeys(sample_cols))
    print("  Example rows (up to 20):")
    sample = collisions[sample_cols].head(20)
    with pd.option_context(
        "display.max_colwidth", 80, "display.width", 180
    ):
        print(sample.to_string(index=False))

    return {
        "check": name,
        "key_col": key_col,
        "n_leaked_groups": int(n_groups),
        "n_leaked_rows": int(n_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to master_index.parquet. Overrides FLAME_MASTER_INDEX env.",
    )
    args = parser.parse_args(argv)

    index_path = _resolve_index_path(args.csv)
    report_path = _resolve_report_path()

    if not index_path.exists():
        print(f"ERROR: index not found at {index_path}", file=sys.stderr)
        print(
            "Hint: pass --csv /path/to/master_index.parquet or set "
            "FLAME_MASTER_INDEX env var.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_parquet(index_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(
            f"ERROR: master_index is missing required columns: {missing}",
            file=sys.stderr,
        )
        return 2

    df = df[REQUIRED_COLUMNS].copy()
    df["path_rgb_norm"] = df["path_rgb"].map(_normalize_path)
    df["path_th_norm"] = df["path_th"].map(_normalize_path)
    df["base_rgb"] = df["path_rgb"].map(_base_id_from_path)
    df["base_th"] = df["path_th"].map(_base_id_from_path)

    print(f"Loaded {len(df)} rows from {index_path}")
    print("Split distribution:")
    print(df["split"].value_counts().to_string())
    print("Source distribution:")
    print(df["source"].value_counts().to_string())

    checks = [
        ("path_rgb", "path_rgb_norm"),
        ("path_th", "path_th_norm"),
        ("key", "key"),
        ("split_group", "split_group"),
        ("video_stem_rgb", "base_rgb"),
        ("video_stem_th", "base_th"),
    ]

    summaries: list[dict] = []
    all_leaks: list[pd.DataFrame] = []

    for name, col in checks:
        coll = _collisions(df, col)
        summary = _summarize(name, col, coll)
        summaries.append(summary)
        if not coll.empty:
            tagged = coll.copy()
            tagged.insert(0, "leakage_type", name)
            tagged.insert(1, "leakage_key", tagged[col])
            all_leaks.append(
                tagged[
                    [
                        "leakage_type",
                        "leakage_key",
                        "split",
                        "source",
                        "label",
                        "key",
                        "split_group",
                        "path_rgb",
                        "path_th",
                    ]
                ]
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if all_leaks:
        report = pd.concat(all_leaks, ignore_index=True)
        report.to_csv(report_path, index=False)
        print(f"\nLeakage report written to: {report_path}  ({len(report)} rows)")
    else:
        empty = pd.DataFrame(
            columns=[
                "leakage_type",
                "leakage_key",
                "split",
                "source",
                "label",
                "key",
                "split_group",
                "path_rgb",
                "path_th",
            ]
        )
        empty.to_csv(report_path, index=False)
        print(f"\nLeakage report written to: {report_path}  (empty)")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for s in summaries:
        print(
            f"  {s['check']:<18s} groups={s['n_leaked_groups']:<6d} "
            f"rows={s['n_leaked_rows']}"
        )

    total_leaks = sum(s["n_leaked_rows"] for s in summaries)
    if total_leaks > 0:
        print(
            f"\nLEAKAGE DETECTED: {total_leaks} total leaked rows across "
            f"{sum(1 for s in summaries if s['n_leaked_rows'] > 0)} check(s)."
        )
        return 1

    print("\nNO LEAKAGE FOUND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
