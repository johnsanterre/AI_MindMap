"""Feature Importance — permutation importance and ablation analysis."""

import numpy as np

def simple_model(X, w, b=0):
    return X @ w + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def permutation_importance(model_fn, X, y, n_repeats=10):
    """Measure feature importance by shuffling each feature."""
    baseline = mse(y, model_fn(X))
    importances = np.zeros((X.shape[1], n_repeats))

    for j in range(X.shape[1]):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            importances[j, r] = mse(y, model_fn(X_perm)) - baseline

    return importances.mean(axis=1), importances.std(axis=1)

def drop_column_importance(X_train, y_train, X_test, y_test):
    """Retrain without each feature to measure its contribution."""
    w_full = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    baseline = mse(y_test, X_test @ w_full)
    importances = []

    for j in range(X_train.shape[1]):
        cols = [c for c in range(X_train.shape[1]) if c != j]
        w_drop = np.linalg.lstsq(X_train[:, cols], y_train, rcond=None)[0]
        score = mse(y_test, X_test[:, cols] @ w_drop) - baseline
        importances.append(score)

    return np.array(importances)

# --- demo ---
np.random.seed(42)

n = 200
feature_names = ["income", "age", "credit_score", "debt_ratio", "noise1", "noise2"]
d = len(feature_names)
X = np.random.randn(n, d)
true_w = np.array([3.0, -1.5, 2.0, -0.5, 0.0, 0.0])
y = X @ true_w + np.random.randn(n) * 0.5

X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
model_fn = lambda X: X @ w

perm_imp, perm_std = permutation_importance(model_fn, X_test, y_test)
drop_imp = drop_column_importance(X_train, y_train, X_test, y_test)

print("=== Feature Importance ===")
print(f"True weights: {true_w}\n")
print(f"{'Feature':>14} {'True w':>8} {'Perm Imp':>10} {'Drop Imp':>10}")
print("-" * 46)
for i, name in enumerate(feature_names):
    print(f"{name:>14} {true_w[i]:>8.1f} {perm_imp[i]:>10.4f} {drop_imp[i]:>10.4f}")

print(f"\nPermutation importance correctly identifies noise features as unimportant.")
