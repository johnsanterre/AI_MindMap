"""Regularization — L1 (Lasso), L2 (Ridge), and Elastic Net."""

import numpy as np

def ridge(X, y, alpha=1.0, lr=0.01, epochs=1000):
    """L2 regularization: penalizes sum of squared weights."""
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for _ in range(epochs):
        pred = X @ w + b
        error = pred - y
        w -= lr * ((2/m) * X.T @ error + 2 * alpha * w)
        b -= lr * (2/m) * error.sum()
    return w, b

def lasso(X, y, alpha=1.0, lr=0.01, epochs=1000):
    """L1 regularization: penalizes sum of absolute weights (promotes sparsity)."""
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for _ in range(epochs):
        pred = X @ w + b
        error = pred - y
        w -= lr * ((2/m) * X.T @ error + alpha * np.sign(w))
        b -= lr * (2/m) * error.sum()
    return w, b

# --- demo ---
np.random.seed(42)
# Only features 0,1 matter; features 2-9 are noise
X = np.random.randn(100, 10)
true_w = np.array([3.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_w + np.random.randn(100) * 0.5

print("True weights:", true_w)
print()

w_none, _ = ridge(X, y, alpha=0)
print(f"No reg:  {w_none.round(2)}")

w_ridge, _ = ridge(X, y, alpha=0.1)
print(f"Ridge:   {w_ridge.round(2)}")

w_lasso, _ = lasso(X, y, alpha=0.1)
print(f"Lasso:   {w_lasso.round(2)}")
print(f"\nLasso zeros out noise features: {(np.abs(w_lasso[2:]) < 0.1).all()}")
