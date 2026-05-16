"""Dataframe split helpers used by the trainer.

Two split primitives live here:

* :func:`split_train_val_extra` — pulls a deterministic subset of
  ``source == "extra"`` rows aside as a held-out ``extra_test`` partition and
  returns the remaining rows shuffled by ``random_state``.
* :func:`_group_split_three_way` — splits a dataframe into train / val / test
  using a ``split_group`` column so paired RGB + thermal samples stay together.

Both are imported by :mod:`src.training.trainer`. Behaviour is intentionally
unchanged from the original implementation; this module was previously
shadowed by a duplicate definition that referenced an unimported
``GroupShuffleSplit``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_train_val_extra(
    df: pd.DataFrame,
    extra_test_ratio: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split out a portion of ``source == "extra"`` rows into ``extra_test_df``.

    Returns ``(df_main, extra_test_df)``. ``df_main`` is shuffled deterministically
    with ``random_state``. If there are no ``extra`` rows or the ratio is zero,
    the input is returned unchanged with an empty ``extra_test_df``.
    """
    if df.empty or "source" not in df.columns:
        return df, pd.DataFrame(columns=df.columns)

    extra = df[df["source"].astype(str) == "extra"].copy()
    non_extra = df[df["source"].astype(str) != "extra"].copy()
    if extra.empty or extra_test_ratio <= 0:
        return df, pd.DataFrame(columns=df.columns)

    rng = np.random.default_rng(int(random_state))
    idx = np.arange(len(extra))
    rng.shuffle(idx)
    n_test = int(round(len(extra) * float(extra_test_ratio)))
    n_test = max(0, min(len(extra), n_test))
    test_extra = extra.iloc[idx[:n_test]].copy()
    train_extra = extra.iloc[idx[n_test:]].copy()
    df_main = pd.concat([non_extra, train_extra], ignore_index=True)
    df_main = df_main.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    return df_main, test_extra.reset_index(drop=True)


def _group_split_three_way(
    df: pd.DataFrame,
    test_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-wise three-way split using the ``split_group`` column.

    A ``split_group`` value is assigned to exactly one of train / val / test so
    paired samples (e.g. video frames belonging to the same clip) cannot leak
    across splits. If the column is missing, each row becomes its own group.
    """
    if df.empty:
        return df, df, df
    if "split_group" not in df.columns:
        df = df.copy()
        df["split_group"] = df.index.astype(str)

    groups = df["split_group"].astype(str).unique().tolist()
    rng.shuffle(groups)

    n = len(groups)
    n_test = int(round(n * float(test_ratio)))
    n_val = int(round(n * float(val_ratio)))
    n_test = max(0, min(n, n_test))
    n_val = max(0, min(n - n_test, n_val))

    test_g = set(groups[:n_test])
    val_g = set(groups[n_test : n_test + n_val])
    train_g = set(groups[n_test + n_val :])

    tr = df[df["split_group"].astype(str).isin(train_g)].copy()
    va = df[df["split_group"].astype(str).isin(val_g)].copy()
    te = df[df["split_group"].astype(str).isin(test_g)].copy()
    return (
        tr.reset_index(drop=True),
        va.reset_index(drop=True),
        te.reset_index(drop=True),
    )
