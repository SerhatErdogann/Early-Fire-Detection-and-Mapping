from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_df_existing_paths(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    """
    Drop rows whose required files are missing on disk.

    Expected columns:
    - rgb: `path_rgb`
    - thermal: `path_th` or `path_thermal`
    - fusion: both
    """
    if df.empty:
        return df, 0

    mode = (mode or "fusion").lower()
    out = df.copy()

    def _exists(p) -> bool:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return False
        pp = Path(str(p))
        return pp.exists() and pp.is_file()

    rgb_ok = out["path_rgb"].apply(_exists) if "path_rgb" in out.columns else False
    th_col = "path_th" if "path_th" in out.columns else ("path_thermal" if "path_thermal" in out.columns else None)
    th_ok = out[th_col].apply(_exists) if th_col else False

    if mode == "rgb":
        keep = rgb_ok
    elif mode == "thermal":
        keep = th_ok
    else:
        keep = rgb_ok & th_ok

    drop_n = int((~keep).sum()) if hasattr(keep, "sum") else 0
    out = out[keep].reset_index(drop=True)
    return out, drop_n
