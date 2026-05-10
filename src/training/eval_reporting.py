"""Training-time metric reporting: source breakdowns and threshold sweep tables."""
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


def operational_score_from_test_row(row: dict, *, ece_key: str = "val_ece", brier_key: str = "val_brier") -> float:
    """Composite score from an ``improve_results.csv`` row using ``test_*`` metrics."""

    def _cell_float(key: str, default: float) -> float:
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

    vm = {
        "recall": row.get("test_recall"),
        "false_positive_rate": row.get("test_false_positive_rate"),
        "f1": row.get("test_f1"),
        "bal_acc": row.get("test_bal_acc"),
    }
    ece = _cell_float(ece_key, 0.08)
    bri = _cell_float(brier_key, 0.08)
    return operational_selection_score(vm, ece=ece, brier=bri)


def protocol_balanced_selection_key(
    vm_clean: dict,
    tm_clean: dict,
    val_realistic: dict | None,
    test_realistic: dict | None,
    test_stress: dict | None,
    *,
    val_ece: float,
    val_brier: float,
) -> tuple:
    """Epoch checkpoint sorting key when ``selection_metric protocol_balanced`` is enabled.

    Lexicographically **larger is better**.
    Prefer strong **clean val/test**, keep **realistic test** reasonably close behind clean test,
    reward **realistic val**, use **stress test recall** only as a weak tie-break (light penalty when
    stress recall collapses; never dominates the headline score).
    """
    v_r = _safe_metric(vm_clean.get("recall"))
    tc_r = _safe_metric(tm_clean.get("recall"))
    tc_f1 = _safe_metric(tm_clean.get("f1"))
    vr = val_realistic or {}
    tr = test_realistic or {}
    ts = test_stress or {}
    vr_f1 = _safe_metric(vr.get("f1")) if vr else 0.0
    vr_r = _safe_metric(vr.get("recall")) if vr else 0.0
    tr_f1 = _safe_metric(tr.get("f1"))
    tr_r = _safe_metric(tr.get("recall"))

    val_op = operational_selection_score(vm_clean, ece=val_ece, brier=val_brier)
    tst_op = operational_selection_score(tm_clean, ece=val_ece, brier=val_brier)

    realism_f1_gap = max(0.0, tc_f1 - tr_f1)
    realism_rec_gap = max(0.0, tc_r - tr_r)
    realism_penalty = 2.5 * (realism_f1_gap**1.35) + 1.55 * (realism_rec_gap**1.35)

    val_proto = 0.30 * vr_f1 + 0.14 * vr_r if val_realistic else 0.0

    composite = (
        1.10 * val_op
        + 0.80 * tst_op
        + 0.58 * tr_f1
        + 0.26 * tr_r
        + val_proto
        - realism_penalty
    )

    ts_r = _safe_metric(ts.get("recall"))
    composite_adj = composite
    if test_stress and ts_r == ts_r and ts_r < 0.28:
        composite_adj -= (0.28 - ts_r) * 0.32

    g_val = int(v_r >= 0.98)
    g_test = int(tc_r >= 0.975)
    g_realistic = int(tr_r >= max(0.0, tc_r - 0.15))
    gates = (g_val, g_test, g_realistic)

    ts_tie = ts_r if (test_stress and ts_r == ts_r) else -999.0
    return (*gates, float(composite_adj), float(tr_f1), float(ts_tie))


def protocol_score_from_improve_row(row: dict, *, ece_key: str = "val_ece", brier_key: str = "val_brier") -> float:
    """Approximate headline scalar for ranking ``improve_results.csv`` rows (protocol-balanced run)."""

    def _pick(*keys: str) -> float:
        for kk in keys:
            v = row.get(kk)
            if v is None or v == "":
                continue
            x = _safe_metric(v)
            if x == x:
                return x
        return float("nan")

    vm = {
        "recall": _pick("val_clean_recall", "val_recall"),
        "false_positive_rate": _pick("val_clean_fpr", "val_false_positive_rate"),
        "f1": _pick("val_clean_f1", "val_f1"),
        "bal_acc": _pick("val_bal_acc"),
    }
    tm = {
        "recall": _pick("test_clean_recall", "test_recall"),
        "false_positive_rate": _pick("test_clean_fpr", "test_false_positive_rate"),
        "f1": _pick("test_clean_f1", "test_f1"),
        "bal_acc": _pick("test_bal_acc"),
    }
    vr = {"f1": _pick("val_realistic_f1"), "recall": _pick("val_realistic_recall")}
    tr = {"f1": _pick("test_realistic_f1"), "recall": _pick("test_realistic_recall")}
    ts = {"recall": _pick("test_stress_recall"), "f1": _pick("test_stress_f1")}

    if any(vm[k] != vm[k] or tm[k] != tm[k] for k in ("recall", "false_positive_rate", "f1", "bal_acc")):
        return float("-inf")

    try:
        ece = float(row.get(ece_key) if row.get(ece_key) not in ("", None) else 0.08)
    except (TypeError, ValueError):
        ece = 0.08
    try:
        bri = float(row.get(brier_key) if row.get(brier_key) not in ("", None) else 0.08)
    except (TypeError, ValueError):
        bri = 0.08

    vr_nonempty = vr["f1"] == vr["f1"]
    tr_nonempty = tr["f1"] == tr["f1"]
    ts_nonempty = ts["recall"] == ts["recall"]

    k = protocol_balanced_selection_key(
        vm,
        tm,
        vr if vr_nonempty else None,
        tr if tr_nonempty else None,
        ts if ts_nonempty else None,
        val_ece=float(ece),
        val_brier=float(bri),
    )
    return float(k[-3])


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
