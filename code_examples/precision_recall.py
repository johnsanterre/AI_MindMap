"""Precision-Recall — evaluation metrics for imbalanced classification."""

import numpy as np

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn

def precision(tp, fp):
    return tp / (tp + fp + 1e-8)

def recall(tp, fn):
    return tp / (tp + fn + 1e-8)

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec + 1e-8)

def pr_curve(y_true, scores, n_thresholds=50):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    precisions, recalls = [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
        precisions.append(precision(tp, fp))
        recalls.append(recall(tp, fn))
    return np.array(precisions), np.array(recalls), thresholds

def average_precision(y_true, scores):
    sorted_idx = scores.argsort()[::-1]
    y_sorted = y_true[sorted_idx]
    tp_cumsum = np.cumsum(y_sorted)
    prec_at_k = tp_cumsum / np.arange(1, len(y_sorted) + 1)
    return np.sum(prec_at_k * y_sorted) / (y_sorted.sum() + 1e-8)

# --- demo ---
np.random.seed(42)

# imbalanced dataset: 90% negative, 10% positive
n = 200
y_true = np.array([0]*180 + [1]*20)
# model scores
scores = np.random.randn(n) * 0.5
scores[y_true == 1] += 1.5

print("=== Precision-Recall Analysis ===")
print(f"Positive: {(y_true==1).sum()}, Negative: {(y_true==0).sum()}\n")

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 44)
for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
    y_pred = (scores >= t).astype(int)
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred)
    p, r = precision(tp, fp), recall(tp, fn)
    f1 = f1_score(p, r)
    print(f"{t:>10.1f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

ap = average_precision(y_true, scores)
print(f"\nAverage Precision (AP): {ap:.3f}")
print("AP summarizes the PR curve into a single number.")
