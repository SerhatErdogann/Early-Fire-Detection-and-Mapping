import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.training.metrics import metrics_at_threshold, find_best_threshold_f1, _best_threshold_mode
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.training.metrics import metrics_at_threshold, find_best_threshold_f1, _best_threshold_mode


MANUAL_CSV = Path("outputs/manual_review.csv")
OUT_JSON = Path("outputs/calibration.json")


def _map_label(x: str) -> int:
    x = str(x).strip().lower()
    if x in ("yes_fire", "fire"):
        return 1
    if x in ("no_fire", "no fire"):
        return 0
    if x in ("smoke_only", "hot_nonfire", "uncertain", "unknown"):
        return -1
    return -1


def _best_threshold_for_mode(scores: np.ndarray, labels: np.ndarray, mode: str, use_minmax: bool = False):
    scores = scores.astype(np.float32)
    labels = labels.astype(np.int64)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return None

    if use_minmax:
        lo, hi = float(scores.min()), float(scores.max())
        if hi <= lo:
            ts = np.asarray([lo], dtype=np.float32)
        else:
            ts = np.linspace(lo, hi, 401, dtype=np.float32)
    else:
        ts = np.linspace(0.0, 1.0, 401, dtype=np.float32)

    mode = (mode or "balanced").lower()
    best = {"thr": 0.5, "f1": -1.0, "recall": -1.0, "precision": -1.0}

    for t in ts:
        m = metrics_at_threshold(labels, scores, float(t))
        f1, rec, prec = float(m["f1"]), float(m["recall"]), float(m["precision"])
        if mode == "balanced" or mode == "f1":
            key = f1
            tie = (rec, prec)
        elif mode in ("alarm", "recall", "recall_priority"):
            key = rec * 1000 + f1
            tie = (f1, prec)
        elif mode in ("review", "precision", "precision_priority"):
            key = prec * 1000 + f1
            tie = (f1, rec)
        else:
            key = f1
            tie = (rec, prec)

        cur_key = best.get("_k", -1e18)
        if key > cur_key or (key == cur_key and tie > best.get("_tie", (-1, -1))):
            best = {
                "thr": float(t),
                "f1": f1,
                "recall": rec,
                "precision": prec,
                "_k": key,
                "_tie": tie,
            }

    best.pop("_k", None)
    best.pop("_tie", None)
    return best


def main():
    if not MANUAL_CSV.exists():
        raise SystemExit(f"Manual review CSV bulunamadı: {MANUAL_CSV}")

    df = pd.read_csv(MANUAL_CSV)
    if "label" not in df.columns or "prob_fire" not in df.columns:
        raise SystemExit("manual_review.csv içinde 'label' ve 'prob_fire' kolonları yok.")

    df["y"] = df["label"].map(_map_label)
    df = df[df["y"] >= 0].reset_index(drop=True)

    if len(df) == 0:
        raise SystemExit("Etiketli satır yok. Önce 07_ui.py ile etiketleme yap.")

    y = df["y"].to_numpy()
    prob_scores = df["prob_fire"].to_numpy()

    balanced = _best_threshold_for_mode(prob_scores, y, "balanced", use_minmax=False)
    alarm = _best_threshold_for_mode(prob_scores, y, "alarm", use_minmax=False)
    review = _best_threshold_for_mode(prob_scores, y, "review", use_minmax=False)

    thr_f1_line = find_best_threshold_f1(y, prob_scores)
    thr_alarm_line = _best_threshold_mode(y, prob_scores, "alarm")
    thr_review_line = _best_threshold_mode(y, prob_scores, "review")

    best_risk = None
    if "risk_score" in df.columns:
        best_risk_bal = _best_threshold_for_mode(df["risk_score"].to_numpy(), y, "balanced", use_minmax=True)
        best_risk_alarm = _best_threshold_for_mode(df["risk_score"].to_numpy(), y, "alarm", use_minmax=True)
        best_risk_rev = _best_threshold_for_mode(df["risk_score"].to_numpy(), y, "review", use_minmax=True)
        best_risk = {
            "balanced": best_risk_bal,
            "alarm": best_risk_alarm,
            "review": best_risk_rev,
        }

    out = {
        "n_samples": int(len(df)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "thr_prob_fire": balanced["thr"] if balanced else None,
        "metrics_prob_fire": balanced,
        "thr_prob_fire_balanced": balanced["thr"] if balanced else None,
        "thr_prob_fire_alarm": alarm["thr"] if alarm else None,
        "thr_prob_fire_review": review["thr"] if review else None,
        "metrics_prob_fire_balanced": balanced,
        "metrics_prob_fire_alarm": alarm,
        "metrics_prob_fire_review": review,
        "thr_prob_fire_f1_scan": float(thr_f1_line),
        "thr_prob_fire_alarm_scan": float(thr_alarm_line),
        "thr_prob_fire_review_scan": float(thr_review_line),
        "thr_risk_by_mode": best_risk,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ Kalibrasyon sonuçları:")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nKaydedildi: {OUT_JSON}")


if __name__ == "__main__":
    main()
