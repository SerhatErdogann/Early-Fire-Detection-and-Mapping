from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    p = Path("data/master_index.parquet")
    if not p.exists():
        raise SystemExit("Missing data/master_index.parquet")
    df = pd.read_parquet(p)
    print("rows", int(len(df)))

    for col in ("source", "split_group", "label", "path_rgb", "path_th"):
        if col not in df.columns:
            raise SystemExit(f"Missing column: {col}")

    # Source consistency
    sources = sorted(df["source"].astype(str).unique().tolist())
    print("sources", sources)

    expected = {"binary_root", "flame3", "flame_video_nofire"}
    known_optional = {"cart_aux", "flame3_raw_extra"}
    extra = [s for s in sources if s not in expected and s not in known_optional]
    missing = [s for s in sorted(expected) if s not in sources]
    if extra:
        print("[warn] unexpected_sources", extra)
    if missing:
        print("[warn] missing_sources", missing)

    # Ensure no NaN/empty split_group
    sg = df["split_group"].astype(str)
    print("split_group_empty", int((sg.str.len() == 0).sum()))

    # Quick path existence stats (sample-based to keep it fast)
    # We only check that paths are non-empty strings here.
    pr = df["path_rgb"].astype(str)
    pt = df["path_th"].astype(str)
    print("path_rgb_empty", int((pr.str.len() == 0).sum()))
    print("path_th_empty", int((pt.str.len() == 0).sum()))

    # Label distribution by source
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    tab = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    print("\nby_source_label")
    print(tab.to_string())


if __name__ == "__main__":
    main()

