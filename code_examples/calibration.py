"""Probability Calibration — making model confidence match actual accuracy."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def reliability_diagram(y_true, y_prob, n_bins=10):
    """Compute calibration curve."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_accuracies.append(y_true[mask].mean())
        bin_confidences.append(y_prob[mask].mean())
        bin_counts.append(mask.sum())

    return np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    accs, confs, counts = reliability_diagram(y_true, y_prob, n_bins)
    total = counts.sum()
    return np.sum(counts / total * np.abs(accs - confs))

def temperature_scaling(logits, T):
    return softmax(logits / T)

def find_temperature(logits, y_true, T_range=np.arange(0.1, 5.0, 0.1)):
    """Find temperature that minimizes calibration error."""
    best_T, best_ece = 1.0, np.inf
    for T in T_range:
        probs = temperature_scaling(logits, T)
        max_probs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == y_true).astype(float)
        ece = expected_calibration_error(correct, max_probs)
        if ece < best_ece:
            best_ece = ece
            best_T = T
    return best_T

# --- demo ---
np.random.seed(42)

n_classes = 5
n = 500
# overconfident model (typical of deep nets)
logits = np.random.randn(n, n_classes) * 3  # high magnitude → overconfident
y_true = np.random.randint(0, n_classes, n)
# make some predictions correct
for i in range(n):
    if np.random.rand() < 0.6:
        logits[i, y_true[i]] += 3

probs = softmax(logits)
preds = probs.argmax(axis=1)
max_probs = probs.max(axis=1)
correct = (preds == y_true).astype(float)

ece_before = expected_calibration_error(correct, max_probs)

print("=== Probability Calibration ===\n")
print(f"Accuracy: {correct.mean():.1%}")
print(f"Avg confidence: {max_probs.mean():.1%}")
print(f"ECE (before): {ece_before:.4f}\n")

accs, confs, counts = reliability_diagram(correct, max_probs)
print(f"{'Confidence':>11} {'Accuracy':>9} {'Count':>7} {'Gap':>7}")
print("-" * 38)
for a, c, n_b in zip(accs, confs, counts):
    print(f"{c:>11.2f} {a:>9.2f} {n_b:>7} {abs(a-c):>7.3f}")

# temperature scaling
T_opt = find_temperature(logits, y_true)
probs_cal = temperature_scaling(logits, T_opt)
max_probs_cal = probs_cal.max(axis=1)
preds_cal = probs_cal.argmax(axis=1)
correct_cal = (preds_cal == y_true).astype(float)
ece_after = expected_calibration_error(correct_cal, max_probs_cal)

print(f"\nTemperature scaling: T={T_opt:.1f}")
print(f"ECE after: {ece_after:.4f} (was {ece_before:.4f})")
print(f"Avg confidence after: {max_probs_cal.mean():.1%}")
