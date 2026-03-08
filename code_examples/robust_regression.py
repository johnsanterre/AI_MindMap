"""Robust Regression — resistant to outliers via Huber and RANSAC methods."""

import numpy as np

def huber_loss(r, delta=1.0):
    return np.where(np.abs(r) <= delta,
                    0.5 * r**2,
                    delta * (np.abs(r) - 0.5 * delta))

def huber_regression(X, y, delta=1.0, lr=0.01, n_iter=500):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        r = y - (X @ w + b)
        grad = np.where(np.abs(r) <= delta, -r, -delta * np.sign(r))
        w -= lr * (X.T @ grad) / n
        b -= lr * grad.mean()
    return w, b

def ransac(X, y, n_iter=100, threshold=1.0, min_inliers_ratio=0.5):
    """Random Sample Consensus."""
    n, d = X.shape
    best_w, best_b = None, None
    best_inliers = 0

    for _ in range(n_iter):
        # random minimal sample (2 points for line)
        idx = np.random.choice(n, d + 1, replace=False)
        X_s = np.column_stack([X[idx], np.ones(d+1)])
        y_s = y[idx]
        try:
            params = np.linalg.solve(X_s.T @ X_s, X_s.T @ y_s)
        except np.linalg.LinAlgError:
            continue
        w, b = params[:-1], params[-1]

        # count inliers
        residuals = np.abs(y - (X @ w + b))
        inliers = residuals < threshold
        n_inliers = inliers.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            # refit on all inliers
            X_in = np.column_stack([X[inliers], np.ones(n_inliers)])
            params = np.linalg.lstsq(X_in, y[inliers], rcond=None)[0]
            best_w, best_b = params[:-1], params[-1]

    return best_w, best_b, best_inliers

# --- demo ---
np.random.seed(42)

# clean data with outliers
n = 100
X = np.random.randn(n, 1)
y_clean = 3 * X.flatten() + 1 + np.random.randn(n) * 0.5

# add 15% outliers
n_outliers = 15
outlier_idx = np.random.choice(n, n_outliers, replace=False)
y = y_clean.copy()
y[outlier_idx] += np.random.randn(n_outliers) * 20

# OLS
X_aug = np.column_stack([X, np.ones(n)])
w_ols = np.linalg.lstsq(X_aug, y, rcond=None)[0]

# Huber
w_hub, b_hub = huber_regression(X, y, delta=1.0)

# RANSAC
w_ran, b_ran, n_inliers = ransac(X, y, threshold=2.0)

print("=== Robust Regression ===")
print(f"True: y = 3x + 1, with {n_outliers} outliers\n")
print(f"{'Method':>10} {'Slope':>8} {'Intercept':>10} {'MSE (clean)':>12}")
print("-" * 44)
for name, w, b in [("OLS", w_ols[0], w_ols[1]),
                     ("Huber", w_hub[0], b_hub),
                     ("RANSAC", w_ran[0], b_ran)]:
    pred = X.flatten() * w + b
    mse = np.mean((pred - y_clean)**2)
    print(f"{name:>10} {w:>8.3f} {b:>10.3f} {mse:>12.4f}")

print(f"\nTrue slope=3.0, intercept=1.0")
print(f"RANSAC inliers: {n_inliers}/{n}")
