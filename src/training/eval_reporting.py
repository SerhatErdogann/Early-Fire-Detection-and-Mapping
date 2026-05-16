"""Training-time helpers: per-source metrics, JSON sanitation, selection scores."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    brier_score_binary,
    expected_calibration_error,
    metrics_at_threshold,
)


@contextmanager
def _suppress_single_class_warnings():
    """Silence sklearn warnings raised when a slice has only one class.

    The single-class branches in :func:`source_threshold_recommendations`
    intentionally call ``metrics_at_threshold`` on slices where only
    ``no_fire`` is present (e.g. ``flame_video_nofire``) — this triggers the
    ``y_pred contains classes not in y_true`` UserWarning per threshold and
    pollutes the trainer log. The numbers themselves remain correct.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*y_pred contains classes not in y_true.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Recall is ill-defined.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Precision is ill-defined.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*F-score is ill-defined.*",
            category=UserWarning,
        )
        yield


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
    # Suppress per-threshold sklearn warnings raised on single-class or
    # tail-end thresholds; they otherwise spam the trainer log without
    # changing any of the reported numbers.
    with _suppress_single_class_warnings():
        return _source_threshold_recommendations_impl(
            df_like, ys, ps, thresholds=thresholds, min_recall=min_recall, min_samples=min_samples
        )


def _source_threshold_recommendations_impl(
    df_like,
    ys: np.ndarray,
    ps: np.ndarray,
    *,
    thresholds: np.ndarray | list[float] | None = None,
    min_recall: float = 0.98,
    min_samples: int = 20,
) -> dict[str, dict[str, Any]]:
    """Body of :func:`source_threshold_recommendations`. Kept separate so the
    public function can wrap it in a single ``warnings.catch_warnings`` block
    without re-indenting the loop."""
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
            # Single-class slice: only FPR / specificity make sense. Compute
            # them inline (without sklearn) to avoid the per-threshold
            # ``y_pred contains classes not in y_true`` warnings sklearn
            # raises when recall / precision / F1 are ill-defined.
            best_t: float | None = None
            best_fpr: float | None = None
            best_spec: float | None = None
            single_label = int(yy[0])
            for t in thresholds:
                pred = (pp >= float(t)).astype(np.int64)
                if single_label == 0:
                    fp = int((pred == 1).sum())
                    tn = int((pred == 0).sum())
                    fpr_t = fp / max(1, fp + tn)
                    spec_t = tn / max(1, fp + tn)
                else:
                    # Single-class fire slice: FPR is undefined; report 0/1 by convention.
                    fpr_t = float("nan")
                    spec_t = float("nan")
                if np.isnan(fpr_t):
                    continue
                if best_fpr is None or fpr_t < best_fpr:
                    best_fpr = float(fpr_t)
                    best_spec = float(spec_t)
                    best_t = float(t)
            out[s] = {
                "status": "single_class_no_recall",
                "n": n,
                "single_label": single_label,
                "threshold": best_t,
                "fpr": best_fpr,
                "specificity": best_spec,
            }
            continue

        # Two-class slice: full sweep (sklearn warnings suppressed at caller).
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


def _safe_metric(x: Any, default: float = 0.0) -> float:
    """Return a finite float for ``x`` (NaN / non-numeric -> ``default``)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if v != v or v in (float("inf"), float("-inf")) else v


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
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    return obj


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


def operational_selection_score(vm: dict, *, ece: float, brier: float) -> float:
    """Higher is better: emphasise recall and ranking quality, penalise FPR and miscalibration.

    Used for ``--selection_metric recall_fpr`` checkpoint picking and for post-hoc
    ``improve_results.csv`` ranking alongside raw recall/FPR.
    """
    r = _safe_metric(vm.get("recall"))
    fpr = _safe_metric(vm.get("false_positive_rate"))
    f1 = _safe_metric(vm.get("f1"))
    bal = _safe_metric(vm.get("bal_acc"))
    e = _safe_metric(ece)
    br = _safe_metric(brier)
    return 2.2 * r + 1.45 * f1 + 1.05 * bal - 2.75 * fpr - 0.85 * e - 0.42 * br


def recall_fpr_selection_key(vm: dict, *, ece: float = 0.0, brier: float = 0.0) -> tuple:
    """Sorting key for ``recall_fpr`` policy (epoch checkpoint selection).

    Lexicographically higher is better:
    1. ``recall >= 0.98`` (deployment safety gate)
    2. :func:`operational_selection_score` (recall, F1, bal_acc, −FPR, −ECE, −Brier)
    3. raw recall, −FPR, balanced accuracy, F1
    """
    rec = _safe_metric(vm.get("recall"))
    fpr = _safe_metric(vm.get("false_positive_rate"))
    bal = _safe_metric(vm.get("bal_acc"))
    f1 = _safe_metric(vm.get("f1"))
    meets = int(rec >= 0.98)
    op = operational_selection_score(vm, ece=ece, brier=brier)
    return (meets, op, rec, -fpr, bal, f1)


def operational_score_from_improve_realistic_row(row: dict) -> float:
    """Operational composite from ``test_realistic_*`` (and optional calibration columns)."""

    def _cell(key: str, default: float) -> float:
        v = row.get(key)
        if v is None or v == "":
            return default
        try:
            x = float(v)
        except (TypeError, ValueError):
            return default
        if x != x or x in (float("inf"), float("-inf")):
            return default
        return x

    recall = _cell("test_realistic_recall", 0.0)
    fpr = _cell("test_realistic_fpr", 0.0)
    # ``improve_results`` row may omit bal_acc — approximate from recall & FPR proxy.
    bal_proxy = max(0.0, min(1.0, 0.5 * (recall + max(0.0, 1.0 - fpr))))
    vm = {
        "recall": row.get("test_realistic_recall"),
        "false_positive_rate": row.get("test_realistic_fpr"),
        "f1": row.get("test_realistic_f1"),
        "bal_acc": bal_proxy,
    }
    return operational_selection_score(
        vm, ece=_cell("val_ece", 0.08), brier=_cell("val_brier", 0.08)
    )


# Back-compat for scripts that still import the old name.
operational_score_from_test_row = operational_score_from_improve_realistic_row


def balanced_realistic_rank_score(row: dict) -> float:
    """Blend noisy val/test F1 for balanced deployment picks."""
    return 0.45 * _safe_metric(row.get("val_realistic_f1")) + 0.55 * _safe_metric(row.get("test_realistic_f1"))
