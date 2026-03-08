"""Quantile Regression — predicting conditional quantiles, not just means."""

import numpy as np

def quantile_loss(y, y_pred, tau):
    """Pinball loss: asymmetric penalty for quantile estimation."""
    residual = y - y_pred
    return np.mean(np.maximum(tau * residual, (tau - 1) * residual))

def quantile_regression(X, y, tau=0.5, lr=0.01, n_iter=500):
    """Gradient descent for linear quantile regression."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_iter):
        pred = X @ w + b
        residual = y - pred
        grad_w = np.zeros(d)
        grad_b = 0.0
        for i in range(n):
            if residual[i] >= 0:
                grad_w -= tau * X[i]
                grad_b -= tau
            else:
                grad_w -= (tau - 1) * X[i]
                grad_b -= (tau - 1)
        w -= lr * grad_w / n
        b -= lr * grad_b / n

    return w, b

# --- demo ---
np.random.seed(42)

# heteroscedastic data: variance increases with x
n = 200
X = np.sort(np.random.uniform(0, 5, n)).reshape(-1, 1)
noise_scale = 0.5 + X.flatten()
y = 2 * X.flatten() + 1 + np.random.randn(n) * noise_scale

print("=== Quantile Regression ===")
print("y = 2x + 1 + noise (heteroscedastic)\n")

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"{'Quantile':>9} {'Slope':>7} {'Intercept':>10} {'Loss':>8}")
print("-" * 38)
for tau in quantiles:
    w, b = quantile_regression(X, y, tau=tau, lr=0.01, n_iter=1000)
    pred = X @ w + b
    loss = quantile_loss(y, pred, tau)
    print(f"{tau:>9.2f} {w[0]:>7.3f} {b:>10.3f} {loss:>8.4f}")

# coverage check
w_lo, b_lo = quantile_regression(X, y, tau=0.1)
w_hi, b_hi = quantile_regression(X, y, tau=0.9)
pred_lo = X @ w_lo + b_lo
pred_hi = X @ w_hi + b_hi
coverage = np.mean((y >= pred_lo) & (y <= pred_hi))
print(f"\n80% prediction interval coverage: {coverage:.1%}")
print(f"(Expected: 80%)")
print(f"\nQuantile regression captures how uncertainty varies with x.")
