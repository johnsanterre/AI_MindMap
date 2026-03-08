"""Cross Validation — k-fold CV from scratch."""

import numpy as np

def k_fold_split(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds

def simple_linear_regression(X, y):
    X_b = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(X_b, y, rcond=None)[0]
    return w

def predict(X, w):
    X_b = np.column_stack([np.ones(len(X)), X])
    return X_b @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- demo ---
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ [2, -1, 0.5] + 3 + np.random.randn(100) * 0.5

print("5-Fold Cross Validation:\n")
folds = k_fold_split(len(X), k=5)
scores = []
for i, (train_idx, val_idx) in enumerate(folds):
    w = simple_linear_regression(X[train_idx], y[train_idx])
    preds = predict(X[val_idx], w)
    score = mse(y[val_idx], preds)
    scores.append(score)
    print(f"  Fold {i+1}: MSE = {score:.4f}  (train={len(train_idx)}, val={len(val_idx)})")

print(f"\nMean MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
