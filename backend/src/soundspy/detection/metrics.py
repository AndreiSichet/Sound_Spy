"""Detection metrics (per class and macro).

* **AP** (average precision): how well the probabilities rank windows *with* the class above
  windows *without* it, summarised over every possible threshold. 1.0 = perfect ranking; a
  random guess scores about the fraction of positive windows.
* **mAP**: the mean AP over classes (each class counts equally).
* **ROC-AUC**: the chance that a random positive window scores above a random negative one.
* **F1 / precision / recall** at a chosen threshold: precision = how many "yes" answers were
  right, recall = how many real occurrences were found, F1 = their harmonic mean.

Thresholds are tuned **per class on val** (maximising F1) and only then applied to test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _valid(y: np.ndarray) -> bool:
    return 0 < y.sum() < y.shape[0]


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score)) if _valid(y_true) else float("nan")


def prf_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float, float]:
    pred = y_score >= threshold
    tp = float(np.sum(pred & (y_true > 0.5)))
    fp = float(np.sum(pred & (y_true <= 0.5)))
    fn = float(np.sum(~pred & (y_true > 0.5)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 (``(threshold, f1)``); 0.5 if the class never occurs."""
    from sklearn.metrics import precision_recall_curve

    if y_true.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def tune_thresholds(y_true: np.ndarray, y_score: np.ndarray, labels: Sequence[str]
                    ) -> dict[str, float]:
    # per-class columns via the transpose (rows = windows, columns = classes)
    cols_true, cols_score = y_true.T, y_score.T
    return {lab: tune_threshold(cols_true[j], cols_score[j])[0]
            for j, lab in enumerate(labels)}


def evaluate(y_true: np.ndarray, y_score: np.ndarray, labels: Sequence[str],
             thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Per-class AP, ROC-AUC, (P, R, F1 at threshold), support; macro means."""
    per: dict[str, dict[str, float]] = {}
    cols_true, cols_score = y_true.T, y_score.T
    for j, lab in enumerate(labels):
        t, s = cols_true[j], cols_score[j]
        row = {"ap": average_precision(t, s), "roc_auc": roc_auc(t, s),
               "positives": int(t.sum()), "windows": int(t.shape[0])}
        if thresholds is not None:
            th = float(thresholds[lab])
            p, r, f = prf_at(t, s, th)
            row.update({"threshold": th, "precision": p, "recall": r, "f1": f})
        per[lab] = row

    def macro(key: str) -> float:
        vals = [v[key] for v in per.values() if key in v and not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    out: dict[str, Any] = {"per_class": per, "map": macro("ap"), "roc_auc": macro("roc_auc")}
    if thresholds is not None:
        out.update({"f1": macro("f1"), "precision": macro("precision"),
                    "recall": macro("recall")})
    return out


def false_positive_cooccurrence(y_true: np.ndarray, y_score: np.ndarray,
                                thresholds: dict[str, float], labels: Sequence[str]
                                ) -> dict[str, dict[str, int]]:
    """For each predicted class, which *true* classes were present in its false positives.

    ``table[predicted][present] = count``; ``present = "background"`` when no class was there.
    Shows which classes the model confuses.
    """
    table: dict[str, dict[str, int]] = {lab: {} for lab in labels}
    th = np.array([thresholds[lab] for lab in labels])
    pred = y_score >= th
    for i in range(y_true.shape[0]):
        present = [labels[k] for k in np.flatnonzero(y_true[i] > 0.5)] or ["background"]
        for j in np.flatnonzero(pred[i] & (y_true[i] <= 0.5)):
            row = table[labels[j]]
            for name in present:
                row[name] = row.get(name, 0) + 1
    return table
