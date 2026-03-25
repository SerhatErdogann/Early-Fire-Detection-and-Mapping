import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
)


def eval_probs(model, loader, device, temperature=1.0):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            logits_t = logits / max(1e-6, float(temperature))
            prob = torch.softmax(logits_t, dim=1)[:, 1].cpu().numpy()
            ps.extend(prob.tolist())
            ys.extend(y.numpy().tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(ps, dtype=np.float32)


def eval_logits(model, loader, device):
    model.eval()
    ys, logits_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            logits_list.append(logits.cpu().numpy())
            ys.extend(y.numpy().tolist())
    return np.asarray(ys, dtype=np.int64), np.concatenate(logits_list, axis=0)


def fit_temperature(ys: np.ndarray, logits: np.ndarray, T_grid=None):
    if T_grid is None:
        T_grid = np.linspace(0.5, 4.0, 36)
    best_t, best_nll = 1.0, 1e18
    for t in T_grid:
        t = float(t)
        logits_t = logits / t
        log_probs = logits_t - np.log(np.sum(np.exp(logits_t), axis=1, keepdims=True) + 1e-12)
        nll = -np.mean(log_probs[np.arange(len(ys)), ys])
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t


def metrics_at_threshold(ys: np.ndarray, ps: np.ndarray, thr: float):
    pred = (ps >= thr).astype(np.int64)
    acc = accuracy_score(ys, pred)
    auc = roc_auc_score(ys, ps) if len(set(ys.tolist())) == 2 else float("nan")
    ap = average_precision_score(ys, ps) if len(set(ys.tolist())) == 2 else float("nan")
    cm = confusion_matrix(ys, pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        ys, pred, average="binary", zero_division=0
    )
    return {
        "acc": acc,
        "auc": auc,
        "ap": ap,
        "cm": cm,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def find_best_threshold_f1(ys: np.ndarray, ps: np.ndarray):
    ts = np.linspace(0.05, 0.95, 181)
    best = {"thr": 0.5, "f1": -1.0, "recall": -1.0, "precision": -1.0}
    for t in ts:
        m = metrics_at_threshold(ys, ps, float(t))
        f1, rec, prec = float(m["f1"]), float(m["recall"]), float(m["precision"])
        if (
            f1 > best["f1"]
            or (f1 == best["f1"] and rec > best["recall"])
            or (f1 == best["f1"] and rec == best["recall"] and prec > best["precision"])
        ):
            best = {"thr": float(t), "f1": f1, "recall": rec, "precision": prec}
    return best["thr"]


def expected_calibration_error(ys: np.ndarray, ps: np.ndarray, n_bins: int = 15):
    """ECE for binary fire probability (positive class)."""
    ys = np.asarray(ys, dtype=np.int64)
    ps = np.asarray(ps, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(ys)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (ps >= lo) & (ps < hi) if i < n_bins - 1 else (ps >= lo) & (ps <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(ps[m].mean())
        acc = float(ys[m].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def brier_score_binary(ys: np.ndarray, ps: np.ndarray) -> float:
    ys = np.asarray(ys, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    return float(np.mean((ps - ys) ** 2))


def _best_threshold_mode(ys: np.ndarray, ps: np.ndarray, mode: str) -> float:
    ts = np.linspace(0.05, 0.95, 181)
    mode = (mode or "balanced").lower()
    best_thr, best_score = 0.5, -1e18
    for t in ts:
        m = metrics_at_threshold(ys, ps, float(t))
        rec, prec, f1 = float(m["recall"]), float(m["precision"]), float(m["f1"])
        if mode == "recall" or mode == "alarm":
            score = rec * 1000 + f1
        elif mode == "precision" or mode == "review":
            score = prec * 1000 + f1
        else:
            score = f1
        if score > best_score:
            best_score = score
            best_thr = float(t)
    return best_thr
