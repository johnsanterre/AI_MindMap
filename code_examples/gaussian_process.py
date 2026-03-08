"""Gaussian Process — Bayesian nonparametric regression with uncertainty."""

import numpy as np

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    dists = (x1[:, None] - x2[None, :])**2
    return variance * np.exp(-0.5 * dists / length_scale**2)

def gp_predict(X_train, y_train, X_test, length_scale=1.0, noise=0.1):
    K = rbf_kernel(X_train, X_train, length_scale) + noise**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale)
    K_ss = rbf_kernel(X_test, X_test, length_scale)

    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    std = np.sqrt(np.diag(cov).clip(0))
    return mu, std

# --- demo ---
np.random.seed(42)

# true function
f = lambda x: np.sin(2 * x) + 0.5 * np.cos(5 * x)

# training data
X_train = np.array([-2, -1.5, -0.5, 0.5, 1.5, 2.5])
y_train = f(X_train) + np.random.randn(len(X_train)) * 0.1

# test points
X_test = np.linspace(-3, 3, 20)

mu, std = gp_predict(X_train, y_train, X_test, length_scale=0.7, noise=0.1)
y_true = f(X_test)

print("=== Gaussian Process Regression ===")
print(f"Training points: {len(X_train)}, Test points: {len(X_test)}\n")
print(f"{'x':>6} {'True f(x)':>10} {'GP Mean':>10} {'GP Std':>8} {'In 2σ?':>8}")
print("-" * 46)
for i in range(0, len(X_test), 2):
    in_ci = abs(y_true[i] - mu[i]) < 2 * std[i]
    print(f"{X_test[i]:>6.2f} {y_true[i]:>10.3f} {mu[i]:>10.3f} {std[i]:>8.3f} {'yes' if in_ci else 'no':>8}")

rmse = np.sqrt(np.mean((y_true - mu)**2))
coverage = np.mean(np.abs(y_true - mu) < 2 * std)
print(f"\nRMSE: {rmse:.3f}")
print(f"95% CI coverage: {coverage:.0%}")
