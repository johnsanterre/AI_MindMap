"""Classification Metrics — confusion matrix, precision, recall, F1, ROC."""

import numpy as np

def confusion_matrix(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1(y_true, y_pred, pos=1):
    tp = np.sum((y_pred == pos) & (y_true == pos))
    fp = np.sum((y_pred == pos) & (y_true != pos))
    fn = np.sum((y_pred != pos) & (y_true == pos))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def roc_curve(y_true, scores, n_thresholds=50):
    thresholds = np.linspace(scores.max(), scores.min(), n_thresholds)
    tprs, fprs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    return fprs, tprs

def auc(fprs, tprs):
    return np.trapezoid(tprs, fprs)

# --- demo ---
np.random.seed(42)
y_true = np.array([1]*40 + [0]*60)
scores = np.where(y_true == 1, np.random.randn(100) + 1.5, np.random.randn(100))
y_pred = (scores >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
p, r, f1 = precision_recall_f1(y_true, y_pred)
fprs, tprs = roc_curve(y_true, scores)
area = auc(fprs, tprs)

print("Confusion Matrix:")
print(f"  Predicted:  0    1")
print(f"  Actual 0: {cm[0,0]:3d}  {cm[0,1]:3d}")
print(f"  Actual 1: {cm[1,0]:3d}  {cm[1,1]:3d}")
print(f"\nPrecision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"AUC:       {area:.3f}")
