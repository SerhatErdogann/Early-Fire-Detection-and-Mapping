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


def source_threshold_recommendations(
    df_like,
    ys: np.ndarray,
    ps: np.ndarray,
    *,
    thresholds: np.ndarray | list[float] | None = None,
    min_recall: float = 0.98,
    min_samples: int = 20,
) -> dict[str, dict[str, Any]]:
    """Per-source threshold recommendations.

    For each ``source`` we run a small sweep and pick the **lowest false
    positive rate** threshold whose recall is at least ``min_recall``. This is
    the "miss-no-fires while keeping false alarms manageable" policy from the
    user's spec.

    If no threshold meets ``min_recall`` for a source we fall back to the
    threshold that maximises recall (tie-break: minimum FPR). Sources with a
    single class present or fewer than ``min_samples`` rows are reported as
    skipped instead of silently dropped.

    Returns a dict keyed by source name. Each entry contains the chosen
    threshold, its recall / specificity / FPR / F1, and a ``status`` field
    that says whether the recall target was hit, missed (and we fell back),
    or the source was skipped entirely.
    """
    out: dict[str, dict[str, Any]] = {}
    df = df_like
    if df is None or len(df) == 0:
        return out
    if "source" not in df.columns or len(df) != len(ys):
        return out
    if thresholds is None:
        thresholds = np.arange(0.30, 0.901, 0.05)
    src_arr = df["source"].astype(str).to_numpy()
    for s in sorted(set(src_arr.tolist())):
        m = src_arr == s
        n = int(m.sum())
        if n < int(min_samples):
            out[s] = {"status": "skipped_too_few", "n": n}
            continue
        yy = np.asarray(ys[m], dtype=np.int64)
        pp = np.asarray(ps[m], dtype=np.float64)
        if len(set(yy.tolist())) < 2:
            # Single-class slice: report only FPR / specificity (no recall).
            ms_pick = None
            best_t = None
            best_fpr = None
            for t in thresholds:
                ms_t = metrics_at_threshold(yy, pp, float(t))
                fpr_t = float(ms_t.get("false_positive_rate", float("nan")))
                if np.isnan(fpr_t):
                    continue
                if best_fpr is None or fpr_t < best_fpr:
                    best_fpr = fpr_t
                    best_t = float(t)
                    ms_pick = ms_t
            out[s] = {
                "status": "single_class_no_recall",
                "n": n,
                "threshold": best_t,
                "fpr": best_fpr,
                "specificity": float(ms_pick.get("specificity", float("nan"))) if ms_pick else None,
            }
            continue

        # Two-class slice: full sweep.
        rows: list[dict[str, float]] = []
        for t in thresholds:
            ms_t = metrics_at_threshold(yy, pp, float(t))
            rows.append(
                {
                    "threshold": float(t),
                    "recall": float(ms_t.get("recall", 0.0)),
                    "fpr": float(ms_t.get("false_positive_rate", 1.0)),
                    "specificity": float(ms_t.get("specificity", 0.0)),
                    "f1": float(ms_t.get("f1", 0.0)),
                    "precision": float(ms_t.get("precision", 0.0)),
                }
            )
        eligible = [r for r in rows if r["recall"] >= float(min_recall)]
        if eligible:
            chosen = sorted(eligible, key=lambda r: (r["fpr"], -r["f1"]))[0]
            status = "ok"
        else:
            chosen = sorted(rows, key=lambda r: (-r["recall"], r["fpr"]))[0]
            status = "below_recall_target"
        out[s] = {
            "status": status,
            "n": n,
            "threshold": chosen["threshold"],
            "recall": chosen["recall"],
            "fpr": chosen["fpr"],
            "specificity": chosen["specificity"],
            "f1": chosen["f1"],
            "precision": chosen["precision"],
        }
    return out


def threshold_sweep_grid(
    vy: np.ndarray,
    vp: np.ndarray,
    ty: np.ndarray,
    tp: np.ndarray,
    thresholds: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Dense table of metrics at each threshold on val / test splits."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.901, 0.05)
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
        row["realistic_score_val"] = realistic_selection_score(mv)
        row["realistic_score_test"] = realistic_selection_score(mt)
        rows.append(row)
    return pd.DataFrame(rows)


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

    return {
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


def _safe_metric(x: Any, default: float = 0.0) -> float:
    """Return a finite float for ``x`` (NaN / non-numeric -> ``default``)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if v != v or v in (float("inf"), float("-inf")) else v


def realistic_selection_score(vm: dict) -> float:
    """Composite epoch-selection score on the validation split.

    Formula: ``F1 + bal_acc + AP - 0.5 * FPR``.

    - ``F1`` and ``bal_acc`` capture the operating-point quality.
    - ``AP`` (average precision) rewards threshold-independent ranking.
    - ``-0.5 * FPR`` penalises false alarms.

    Only consumed when the trainer is invoked with ``--selection_metric realistic``;
    the default ``f1_balacc`` policy uses the legacy ``0.5 * (F1 + bal_acc)`` score.
    NaN-safe for early epochs that may produce undefined metrics.
    """
    f1 = _safe_metric(vm.get("f1"))
    bal_acc = _safe_metric(vm.get("bal_acc"))
    ap = _safe_metric(vm.get("ap"))
    fpr = _safe_metric(vm.get("false_positive_rate"))
    return f1 + bal_acc + ap - 0.5 * fpr


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
