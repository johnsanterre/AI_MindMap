"""AUC-ROC — Receiver Operating Characteristic for binary classification."""

import numpy as np

def roc_curve(y_true, scores, n_thresholds=200):
    thresholds = np.linspace(scores.max() + 0.1, scores.min() - 0.1, n_thresholds)
    tprs, fprs = [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tprs.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
    return np.array(fprs), np.array(tprs), thresholds

def auc_trapezoidal(fprs, tprs):
    sorted_idx = np.argsort(fprs)
    return np.trapezoid(tprs[sorted_idx], fprs[sorted_idx])

def auc_wilcoxon(y_true, scores):
    """AUC = P(score_positive > score_negative)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    count = sum(1 for p in pos for n in neg if p > n) + \
            0.5 * sum(1 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))

# --- demo ---
np.random.seed(42)

n = 200
y_true = np.array([0]*100 + [1]*100)

print("=== AUC-ROC Analysis ===\n")

models = {
    "Perfect":   np.concatenate([np.zeros(100), np.ones(100)]),
    "Good":      np.concatenate([np.random.randn(100), np.random.randn(100) + 2]),
    "Weak":      np.concatenate([np.random.randn(100), np.random.randn(100) + 0.5]),
    "Random":    np.random.randn(200),
}

print(f"{'Model':>10} {'AUC (trap)':>11} {'AUC (Wilcoxon)':>15}")
print("-" * 40)
for name, scores in models.items():
    fprs, tprs, _ = roc_curve(y_true, scores)
    auc_t = auc_trapezoidal(fprs, tprs)
    auc_w = auc_wilcoxon(y_true, scores)
    print(f"{name:>10} {auc_t:>11.3f} {auc_w:>15.3f}")

# ROC curve for "Good" model
print(f"\n--- ROC Curve (Good model) ---")
fprs, tprs, thresholds = roc_curve(y_true, models["Good"])
print(f"{'FPR':>6} {'TPR':>6} {'Threshold':>10}")
for i in range(0, len(fprs), 25):
    print(f"{fprs[i]:>6.3f} {tprs[i]:>6.3f} {thresholds[i]:>10.3f}")

# optimal threshold (Youden's J)
j = tprs - fprs
best_idx = j.argmax()
print(f"\nOptimal threshold: {thresholds[best_idx]:.3f} (TPR={tprs[best_idx]:.3f}, FPR={fprs[best_idx]:.3f})")
print(f"AUC = probability that positive scores higher than negative.")
