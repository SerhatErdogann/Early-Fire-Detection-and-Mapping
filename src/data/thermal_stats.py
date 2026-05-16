"""Utilities for thermal channel global statistics (training-set based)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import read_thermal_raw


def estimate_thermal_mu_sigma(
    df,
    *,
    path_col: str = "path_th",
    max_samples: int = 500,
    rng=None,
) -> tuple[float, float]:
    """Welford streaming mean / std on finite raw thermal pixels (subsampling rows)."""
    df_r = df.reset_index(drop=True)
    if path_col not in df_r.columns and "path_thermal" in df_r.columns:
        path_col = "path_thermal"
    if path_col not in df_r.columns or len(df_r) == 0:
        return 0.0, 1.0
    rng = rng or np.random.default_rng()
    idx = rng.choice(len(df_r), size=min(int(max_samples), len(df_r)), replace=False)
    count = 0
    mean = 0.0
    M2 = 0.0
    for ii in idx:
        p = Path(str(df_r.iloc[int(ii)][path_col]))
        if not p.exists():
            continue
        raw, _kind = read_thermal_raw(p)
        v = raw[np.isfinite(raw)]
        if v.size == 0:
            continue
        flat = v.reshape(-1)
        step = max(8192, min(65536, len(flat)))
        for s in range(0, len(flat), step):
            chunk = flat[s : s + step].astype(np.float64)
            for val in chunk:
                count += 1
                delta = float(val) - mean
                mean += delta / count
                M2 += delta * (float(val) - mean)
    if count < 2:
        return mean, max(1.0, abs(mean) * 0.1 + 1e-6)
    var = M2 / (count - 1)
    std = float(max(np.sqrt(max(var, 1e-12)), 1e-6))
    return float(mean), float(std)


def thermal_global_zscore_to_01(raw: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Map raw thermal to approx [0,1] via z-score clipped to +-3sigma."""
    x = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.any():
        fill = float(np.median(x[finite]))
    else:
        fill = 0.0
    x = np.where(finite, x, fill).astype(np.float32)
    z = (x - float(mu)) / float(max(sigma, 1e-6))
    z = np.clip(z, -3.0, 3.0)
    return ((z + 3.0) / 6.0).astype(np.float32)
