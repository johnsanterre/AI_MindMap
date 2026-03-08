"""Polynomial Regression — fitting curves of varying degree."""

import numpy as np

def poly_features(x, degree):
    return np.column_stack([x**i for i in range(degree+1)])

def fit(x, y, degree):
    X = poly_features(x, degree)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

def predict(x, w):
    X = poly_features(x, len(w)-1)
    return X @ w

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

# --- demo ---
np.random.seed(42)
x = np.linspace(-3, 3, 30)
y_true = 0.5 * x**3 - 2 * x + 1
y = y_true + np.random.randn(30) * 3

x_test = np.linspace(-3, 3, 100)
y_test = 0.5 * x_test**3 - 2 * x_test + 1

print("True function: 0.5x³ - 2x + 1\n")
print(f"{'Degree':>7} {'Train MSE':>10} {'Test MSE':>10} {'Coefficients'}")
print("-" * 60)

for deg in [1, 2, 3, 5, 10, 15]:
    w = fit(x, y, deg)
    train_mse = mse(y, predict(x, w))
    test_mse = mse(y_test, predict(x_test, w))
    coef_str = ", ".join(f"{c:.2f}" for c in w[:4])
    if len(w) > 4:
        coef_str += ", ..."
    print(f"{deg:>7} {train_mse:>10.3f} {test_mse:>10.3f}  [{coef_str}]")
