"""Coordinate Descent — optimizing one variable at a time."""

import numpy as np

def coordinate_descent(f, grad_i, x0, n_dims, lr=0.01, n_iter=200):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(n_iter):
        for i in range(n_dims):
            g = grad_i(x, i)
            x[i] -= lr * g
        history.append(x.copy())
    return x, np.array(history)

def lasso_coordinate_descent(X, y, lam=0.1, n_iter=100):
    """Coordinate descent for Lasso: min ||y - Xw||² + λ||w||₁."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(n_iter):
        for j in range(d):
            r = y - X @ w + X[:, j] * w[j]  # partial residual
            z = X[:, j] @ r / n
            w[j] = np.sign(z) * max(abs(z) - lam, 0) / (np.sum(X[:, j]**2) / n)
    return w

# --- demo ---
np.random.seed(42)

# Lasso with sparse true weights
n, d = 100, 20
X = np.random.randn(n, d)
w_true = np.zeros(d)
w_true[:5] = [3, -2, 1.5, -1, 0.5]
y = X @ w_true + np.random.randn(n) * 0.5

print("=== Coordinate Descent for Lasso ===")
print(f"n={n}, d={d}, true nonzeros: 5\n")

print(f"{'λ':>6} {'Nonzeros':>9} {'MSE':>10} {'||w-w*||':>10}")
print("-" * 38)
for lam in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    w = lasso_coordinate_descent(X, y, lam=lam)
    nonzeros = np.sum(np.abs(w) > 0.01)
    mse = np.mean((y - X @ w)**2)
    err = np.linalg.norm(w - w_true)
    print(f"{lam:>6.3f} {nonzeros:>9} {mse:>10.4f} {err:>10.4f}")

# show recovered weights at best lambda
w_best = lasso_coordinate_descent(X, y, lam=0.05)
print(f"\nTrue weights (first 8): {w_true[:8].round(2)}")
print(f"Lasso (λ=0.05):        {w_best[:8].round(2)}")
