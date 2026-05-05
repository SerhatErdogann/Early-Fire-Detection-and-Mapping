"""Training-time metric reporting: source breakdowns and threshold sweep tables."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    brier_score_binary,
    expected_calibration_error,
    metrics_at_threshold,
)


def _jsonable_metric_row(m: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in m.items():
        if k == "cm" and hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
            except Exception:
                out[k] = float(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def metrics_per_source(
    df_like,
    ys: np.ndarray,
    ps: np.ndarray,
    thr: float,
    *,
    min_samples: int = 1,
) -> dict[str, dict]:
    """
    Evaluate binary metrics per ``source`` for rows aligned with ``ys`` / ``ps``.

    Drops sources with fewer than ``min_samples`` rows or fewer than two classes.
    """
    df = df_like
    out: dict[str, dict[str, Any]] = {}
    if df is None or len(df) == 0:
        return out
    if "source" not in df.columns or len(df) != len(ys):
        return out
    src_arr = df["source"].astype(str).to_numpy()
    for s in sorted(set(src_arr.tolist())):
        m = src_arr == s
        n = int(m.sum())
        if n < int(min_samples):
            continue
        yy = np.asarray(ys[m], dtype=np.int64)
        pp = np.asarray(ps[m], dtype=np.float64)
        if len(set(yy.tolist())) < 2:
            continue
        ms = metrics_at_threshold(yy, pp, float(thr))
        row = {"n": n, **_jsonable_metric_row(ms)}
        try:
            row["ece"] = float(expected_calibration_error(yy, pp))
        except Exception:
            row["ece"] = None
        try:
            row["brier"] = float(brier_score_binary(yy, pp))
        except Exception:
            row["brier"] = None
        out[s] = row
    return out


def threshold_sweep_grid(
    vy: np.ndarray,
    vp: np.ndarray,
    ty: np.ndarray,
    tp: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
    extra_y_neg: np.ndarray | None = None,
    extra_p_neg: np.ndarray | None = None,
) -> pd.DataFrame:
    """Dense table of metrics at each threshold on val / test splits.

    If ``extra_y_neg`` / ``extra_p_neg`` are provided (typically label==0 subset of external / extra_test rows),
    add ``extra_test_neg_false_positive_rate`` per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.901, 0.05)
    use_ext = (
        extra_y_neg is not None
        and extra_p_neg is not None
        and len(extra_y_neg) == len(extra_p_neg)
        and len(extra_y_neg) > 0
    )
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        t = float(t)
        mv = metrics_at_threshold(vy, vp, t)
        mt = metrics_at_threshold(ty, tp, t)
        row: dict[str, Any] = {
            "threshold": t,
            "val_acc": mv["acc"],
            "val_f1": mv["f1"],
            "val_recall": mv["recall"],
            "val_precision": mv["precision"],
            "val_bal_acc": mv["bal_acc"],
            "val_specificity": mv["specificity"],
            "val_false_positive_rate": mv["false_positive_rate"],
            "test_acc": mt["acc"],
            "test_f1": mt["f1"],
            "test_recall": mt["recall"],
            "test_precision": mt["precision"],
            "test_bal_acc": mt["bal_acc"],
            "test_specificity": mt["specificity"],
            "test_false_positive_rate": mt["false_positive_rate"],
        }
        row["realistic_score_val"] = float(mv["f1"]) + float(mv["bal_acc"]) - 0.5 * float(mv["false_positive_rate"])
        row["realistic_score_test"] = float(mt["f1"]) + float(mt["bal_acc"]) - 0.5 * float(mt["false_positive_rate"])
        if use_ext:
            mm = metrics_at_threshold(np.asarray(extra_y_neg), np.asarray(extra_p_neg), t)
            row["extra_test_neg_false_positive_rate"] = float(mm["false_positive_rate"])
        rows.append(row)
    return pd.DataFrame(rows)


def policy_external_low_false_alarm(
    grid: pd.DataFrame,
    *,
    min_val_recall: float = 0.90,
) -> dict[str, Any] | None:
    """
    Threshold on validation with ``val_recall >= min_val_recall`` that minimizes external / extra_test
    negative FPR (column ``extra_test_neg_false_positive_rate``), tie-break by ``val_f1``.
    """
    if grid is None or len(grid) == 0 or "extra_test_neg_false_positive_rate" not in grid.columns:
        return None
    cand = grid[grid["val_recall"] >= float(min_val_recall)]
    if len(cand) == 0:
        return None
    row = (
        cand.sort_values(
            by=["extra_test_neg_false_positive_rate", "val_f1"],
            ascending=[True, False],
        )
        .iloc[0]
    )
    t = float(row["threshold"])

    def pack(series: pd.Series, prefix: str) -> dict[str, Any]:
        return {
            "threshold": float(series["threshold"]),
            "acc": float(series[f"{prefix}_acc"]),
            "f1": float(series[f"{prefix}_f1"]),
            "recall": float(series[f"{prefix}_recall"]),
            "precision": float(series[f"{prefix}_precision"]),
            "bal_acc": float(series[f"{prefix}_bal_acc"]),
            "specificity": float(series[f"{prefix}_specificity"]),
            "false_positive_rate": float(series[f"{prefix}_false_positive_rate"]),
            "realistic_score": (
                float(series[f"{prefix}_f1"]) + float(series[f"{prefix}_bal_acc"])
                - 0.5 * float(series[f"{prefix}_false_positive_rate"])
            ),
        }

    ev_raw = row.get("extra_test_neg_false_positive_rate")
    try:
        evf = float(ev_raw)
    except (TypeError, ValueError):
        evf = None
    import math as _math

    if evf is not None and (_math.isnan(evf) or _math.isinf(evf)):
        evf = None
    return {
        "threshold": t,
        "min_val_recall_constraint": float(min_val_recall),
        "strategy": (
            "val_recall>="
            + str(min_val_recall)
            + ", minimize extra_test_neg_false_positive_rate, tie_break val_f1"
        ),
        "val": pack(row, "val"),
        "test": pack(row, "test"),
        "extra_eval": {
            "extra_test_negative_false_positive_rate_at_threshold": evf,
        },
    }


def select_threshold_policies(grid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Heuristic policies on the sweep table (threshold chosen on **val** metrics).

    - high_recall: maximize val recall, tie-break maximize val specificity
    - balanced: maximize val F1
    - low_false_alarm: minimize val FPR, tie-break maximize val_f1
    """
    if grid is None or len(grid) == 0:
        return {}

    hr = grid.sort_values(by=["val_recall", "val_specificity"], ascending=[False, False]).iloc[0]
    balanced = grid.sort_values(by=["val_f1", "realistic_score_val"], ascending=[False, False]).iloc[0]
    lfa = grid.sort_values(by=["val_false_positive_rate", "val_f1"], ascending=[True, False]).iloc[0]

    def pack(row: pd.Series, split: str) -> dict[str, Any]:
        t = float(row["threshold"])
        if split == "val":
            return {
                "threshold": t,
                "acc": float(row["val_acc"]),
                "f1": float(row["val_f1"]),
                "recall": float(row["val_recall"]),
                "precision": float(row["val_precision"]),
                "bal_acc": float(row["val_bal_acc"]),
                "specificity": float(row["val_specificity"]),
                "false_positive_rate": float(row["val_false_positive_rate"]),
                "realistic_score": float(row["realistic_score_val"]),
            }
        return {
            "threshold": t,
            "acc": float(row["test_acc"]),
            "f1": float(row["test_f1"]),
            "recall": float(row["test_recall"]),
            "precision": float(row["test_precision"]),
            "bal_acc": float(row["test_bal_acc"]),
            "specificity": float(row["test_specificity"]),
            "false_positive_rate": float(row["test_false_positive_rate"]),
            "realistic_score": float(row["realistic_score_test"]),
        }

    out = {
        "high_recall": {
            "threshold": float(hr["threshold"]),
            "val": pack(hr, "val"),
            "test": pack(hr, "test"),
            "strategy": "max val_recall, tie_break val_specificity",
        },
        "balanced": {
            "threshold": float(balanced["threshold"]),
            "val": pack(balanced, "val"),
            "test": pack(balanced, "test"),
            "strategy": "max val_f1",
        },
        "low_false_alarm": {
            "threshold": float(lfa["threshold"]),
            "val": pack(lfa, "val"),
            "test": pack(lfa, "test"),
            "strategy": "min val_false_positive_rate, tie_break val_f1",
        },
    }
    ext = policy_external_low_false_alarm(grid, min_val_recall=0.90)
    if ext:
        out["external_low_false_alarm"] = ext
    return out


def realistic_selection_score(vm: dict) -> float:
    """val: f1 + bal_acc - 0.5 * fpr (composite for epoch selection)."""
    return float(vm["f1"]) + float(vm["bal_acc"]) - 0.5 * float(vm["false_positive_rate"])


def sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for JSON dumping."""
    import math

    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    return obj
