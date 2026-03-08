"""Class Weights — handling imbalanced data by adjusting loss weights."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def weighted_cross_entropy(logits, y_onehot, class_weights):
    probs = softmax(logits)
    log_probs = np.log(probs + 1e-8)
    weighted = y_onehot * log_probs * class_weights
    return -np.mean(np.sum(weighted, axis=1))

def compute_class_weights(y, strategy='balanced'):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    if strategy == 'balanced':
        weights = n / (len(classes) * counts)
    elif strategy == 'sqrt':
        weights = np.sqrt(n / (len(classes) * counts))
    else:
        weights = np.ones(len(classes))
    return weights

def train_with_weights(X, y, class_weights, lr=0.1, n_epochs=200):
    n, d = X.shape
    k = len(np.unique(y))
    y_oh = np.eye(k)[y]
    W = np.random.randn(d, k) * 0.1
    for _ in range(n_epochs):
        logits = X @ W
        probs = softmax(logits)
        sample_weights = class_weights[y]
        grad = X.T @ ((probs - y_oh) * sample_weights[:, None]) / n
        W -= lr * grad
    return W

# --- demo ---
np.random.seed(42)

# highly imbalanced: 90% class 0, 8% class 1, 2% class 2
n = 500
y = np.array([0]*450 + [1]*40 + [2]*10)
d = 5
X = np.random.randn(n, d)
# class-specific signal
X[y == 0] += [0.5, 0, 0, 0, 0]
X[y == 1] += [0, 0.5, 0, 0, 0]
X[y == 2] += [0, 0, 0.5, 0, 0]

print("=== Class Weights for Imbalanced Data ===")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

for strategy in ['none', 'balanced', 'sqrt']:
    weights = compute_class_weights(y, strategy)
    if strategy == 'none':
        weights = np.ones(3)
    W = train_with_weights(X, y, weights)
    preds = softmax(X @ W).argmax(axis=1)

    print(f"--- Strategy: {strategy} ---")
    print(f"  Weights: {weights.round(2)}")
    for c in range(3):
        recall = (preds[y == c] == c).mean() if (y == c).sum() > 0 else 0
        print(f"  Class {c} recall: {recall:.1%} ({(y==c).sum()} samples)")
    overall = (preds == y).mean()
    print(f"  Overall accuracy: {overall:.1%}\n")

print("Without weights: model ignores rare classes.")
print("Balanced weights force equal attention to each class.")
