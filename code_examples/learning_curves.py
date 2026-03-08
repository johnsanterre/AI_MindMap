"""Learning Curves — diagnosing bias vs variance from training set size."""

import numpy as np

def polynomial_features(X, degree):
    return np.column_stack([X**i for i in range(degree + 1)])

def ridge_fit(X, y, lam=0.01):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def learning_curve(X_train, y_train, X_val, y_val, degree, sizes):
    train_errors, val_errors = [], []
    for n in sizes:
        X_sub = polynomial_features(X_train[:n], degree)
        w = ridge_fit(X_sub, y_train[:n])
        train_pred = X_sub @ w
        val_pred = polynomial_features(X_val, degree) @ w
        train_errors.append(mse(y_train[:n], train_pred))
        val_errors.append(mse(y_val, val_pred))
    return train_errors, val_errors

# --- demo ---
np.random.seed(42)

# true function: sin(x) + noise
X = np.sort(np.random.uniform(-3, 3, 100))
y = np.sin(X) + np.random.randn(100) * 0.3
X_val = np.linspace(-3, 3, 50)
y_val = np.sin(X_val) + np.random.randn(50) * 0.3

sizes = [5, 10, 20, 40, 60, 80]

print("=== Learning Curves: Bias vs Variance ===\n")

for degree, label in [(1, "Underfitting (deg=1)"), (4, "Good fit (deg=4)"), (15, "Overfitting (deg=15)")]:
    train_err, val_err = learning_curve(X, y, X_val, y_val, degree, sizes)
    print(f"--- {label} ---")
    print(f"{'N':>5} {'Train MSE':>10} {'Val MSE':>10} {'Gap':>10}")
    for i, n in enumerate(sizes):
        gap = val_err[i] - train_err[i]
        print(f"{n:>5} {train_err[i]:>10.4f} {val_err[i]:>10.4f} {gap:>10.4f}")
    print()

print("High bias: both errors high, small gap")
print("High variance: low train error, high val error, large gap")
