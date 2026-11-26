"""Metrics for the M0.5 decision harness.

No scipy/sklearn dependency: Wilson intervals, stratified bootstrap, exact
McNemar, PR-AUC and ROC-AUC are implemented here.  All functions take plain
Python lists / numpy arrays and return JSON-serializable values.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

EPS = 1e-12


def _as_arrays(labels: Sequence[str], answers: Sequence[str], options: Sequence[str]):
    index = {o: i for i, o in enumerate(options)}
    y = np.asarray([index[v] for v in labels], dtype=np.int64)
    pred = np.asarray([index[v] for v in answers], dtype=np.int64)
    return y, pred


def confusion_matrix(y: np.ndarray, pred: np.ndarray, k: int) -> list[list[int]]:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def mcc(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Multiclass Matthews correlation (Gorodkin); reduces to binary MCC."""
    cm = np.asarray(confusion_matrix(y, pred, k), dtype=float)
    n = cm.sum()
    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    correct = np.diag(cm).sum()
    numerator = correct * n - np.sum(true_counts * pred_counts)
    denom = math.sqrt(
        max(n * n - np.sum(pred_counts ** 2), 0.0)
        * max(n * n - np.sum(true_counts ** 2), 0.0)
    )
    return float(numerator / denom) if denom else 0.0


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision for the positive class (y=1), tie-aware by stable sort."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    # AP = sum over positive items of precision at that rank (equivalent to
    # step-wise area under the precision-recall curve, no interpolation).
    return float(np.sum(precision[y == 1]) / y.sum())


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U with average ranks for ties."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def recall_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> dict:
    """Fraud recall at the largest threshold whose normal FPR is <= max_fpr.

    With small normal counts the achievable FPR grid is coarse; return the
    achieved FPR alongside the recall and threshold.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"max_fpr": max_fpr, "threshold": None, "fpr": None, "recall": None}
    # Candidate thresholds: just above each distinct score, descending.
    uniq = np.unique(score)[::-1]
    best = None
    for t in uniq:
        pred = score >= t
        fpr = float(pred[y == 0].mean())
        if fpr <= max_fpr + 1e-12:
            recall = float(pred[y == 1].mean())
            best = {"max_fpr": max_fpr, "threshold": float(t), "fpr": fpr, "recall": recall}
            break
    if best is None:
        best = {"max_fpr": max_fpr, "threshold": float(uniq[-1]), "fpr": 0.0, "recall": 0.0}
    return best


def summarize_binary(
    labels: Sequence[str],
    answers: Sequence[str],
    probs: Sequence[Sequence[float]],
    options: Sequence[str],
    *,
    fpr_targets: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict:
    """Binary summary; assumes options[0] = negative, options[1] = fraud."""
    if len(options) != 2 or list(options)[1] != "涉诈":
        raise ValueError(
            f"binary summary expects [negative, '涉诈'], got {list(options)}"
        )
    negative_name = list(options)[0]
    y, pred = _as_arrays(labels, answers, options)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    n = len(y)
    pos = y == 1
    neg = ~pos

    tp = int(((pred == 1) & pos).sum())
    fn = int(((pred == 0) & pos).sum())
    tn = int(((pred == 0) & neg).sum())
    fp = int(((pred == 1) & neg).sum())
    fraud_recall = _safe_div(tp, tp + fn)
    normal_recall = _safe_div(tn, tn + fp)
    fraud_precision = _safe_div(tp, tp + fp)
    fraud_f1 = _safe_div(2 * fraud_precision * fraud_recall, fraud_precision + fraud_recall)
    acc = _safe_div(tp + tn, n)
    lo, hi = wilson_ci(tp, tp + fn)
    nlo, nhi = wilson_ci(tn, tn + fp)
    plo, phi = wilson_ci(tp, tp + fp)

    out = {
        "n": n,
        "support": {negative_name: int(neg.sum()), "涉诈": int(pos.sum())},
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": acc,
        "balanced_accuracy": (fraud_recall + normal_recall) / 2.0,
        "fraud_recall": fraud_recall,
        "fraud_recall_ci95": [lo, hi],
        "normal_recall_tnr": normal_recall,
        "normal_recall_ci95": [nlo, nhi],
        "fraud_precision": fraud_precision,
        "fraud_precision_ci95": [plo, phi],
        "fraud_f1": fraud_f1,
        "macro_tpr": (fraud_recall + normal_recall) / 2.0,
        "mcc": mcc(y, pred, 2),
        "predicted": {negative_name: int((pred == 0).sum()), "涉诈": int((pred == 1).sum())},
        "mean_confidence": float(conf.mean()),
        "ece_10bin": ece(conf, correct, 10),
        "brier": brier(y, p, 2),
        "nll": nll(y, p),
        "pr_auc": average_precision(y, p[:, 1]),
        "roc_auc": roc_auc(y, p[:, 1]),
        "recall_at_fpr": [recall_at_fpr(y, p[:, 1], t) for t in fpr_targets],
        "confidence_histogram": _confidence_histogram(conf, correct),
    }
    return out


def summarize_multiclass(
    labels: Sequence[str],
    answers: Sequence[str],
    probs: Sequence[Sequence[float]],
    options: Sequence[str],
) -> dict:
    y, pred = _as_arrays(labels, answers, options)
    p = np.asarray(probs, dtype=float)
    k = len(options)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    cm = np.asarray(confusion_matrix(y, pred, k), dtype=np.int64)
    per_class = {}
    for j, option in enumerate(options):
        tp = int(cm[j, j])
        support = int(cm[j, :].sum())
        predicted = int(cm[:, j].sum())
        per_class[option] = {
            "support": support,
            "predicted": predicted,
            "recall": _safe_div(tp, support),
            "recall_ci95": list(wilson_ci(tp, support)),
            "precision": _safe_div(tp, predicted),
        }
    recalls = [v["recall"] for v in per_class.values() if v["support"] > 0]
    f1s = []
    for v in per_class.values():
        if v["support"] > 0:
            r, pr = v["recall"], v["precision"]
            f1s.append(_safe_div(2 * pr * r, pr + r))
    return {
        "n": len(y),
        "accuracy": _safe_div(int((pred == y).sum()), len(y)),
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "options": list(options),
        "mean_confidence": float(conf.mean()),
        "ece_10bin": ece(conf, correct, 10),
        "brier": brier(y, p, k),
        "nll": nll(y, p),
        "confidence_histogram": _confidence_histogram(conf, correct),
    }


def _confidence_histogram(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    return out


def stratified_bootstrap(
    labels: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    seed: int = 20261103,
) -> tuple[float, float, float]:
    """Percentile CI for a statistic over row indices, stratified by class.

    `statistic` receives an index array and returns a float.  The point
    estimate is computed on all rows.  Returns (point, lo, hi).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    idx_by_class = [np.flatnonzero(labels == c) for c in classes]
    point = float(statistic(np.arange(len(labels))))
    reps = []
    for _ in range(n_boot):
        parts = [rng.choice(idx, size=len(idx), replace=True) for idx in idx_by_class]
        idx = np.concatenate(parts)
        value = statistic(idx)
        if not math.isnan(value):
            reps.append(value)
    if not reps:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    labels: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    options: Sequence[str],
    *,
    metric: str = "accuracy",
    n_boot: int = 2000,
    seed: int = 20261103,
) -> dict:
    """Stratified paired bootstrap for metric(A) - metric(B)."""
    index = {o: i for i, o in enumerate(options)}
    y = np.asarray([index[v] for v in labels], dtype=int)
    a = np.asarray([index[v] for v in pred_a], dtype=int)
    b = np.asarray([index[v] for v in pred_b], dtype=int)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_by_class = [np.flatnonzero(y == c) for c in classes]

    def value(sel: np.ndarray) -> float:
        yy = y[sel]
        if metric == "accuracy":
            return float((a[sel] == yy).mean() - (b[sel] == yy).mean())
        if metric == "macro_tpr":
            def mtpr(pred):
                recalls = [float((pred[yy == c] == c).mean()) for c in classes if (yy == c).any()]
                return float(np.mean(recalls))
            return mtpr(a[sel]) - mtpr(b[sel])
        raise ValueError(metric)

    point = value(np.arange(len(y)))
    reps = []
    for _ in range(n_boot):
        parts = [rng.choice(idx, size=len(idx), replace=True) for idx in idx_by_class]
        reps.append(value(np.concatenate(parts)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"metric": metric, "point": point, "ci95": [float(lo), float(hi)]}


def mcnemar_exact(pred_a: Sequence[str], pred_b: Sequence[str],
                  labels: Sequence[str]) -> dict:
    """Exact two-sided McNemar on discordant pairs; no continuity correction."""
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    y = np.asarray(labels)
    a_ok, b_ok = a == y, b == y
    n01 = int((~a_ok & b_ok).sum())  # A wrong, B right
    n10 = int((a_ok & ~b_ok).sum())  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "p_value": min(1.0, 2.0 * tail)}


def flip_rate(forward_answers: Sequence[str], reversed_answers: Sequence[str]) -> dict:
    flips = sum(int(a != b) for a, b in zip(forward_answers, reversed_answers))
    n = len(forward_answers)
    return {"n": n, "flips": flips, "rate": _safe_div(flips, n),
            "ci95": list(wilson_ci(flips, n))}
