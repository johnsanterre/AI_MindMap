"""Feature Scaling — standardization, min-max, robust scaling."""

import numpy as np

def standardize(X):
    """Zero mean, unit variance."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def min_max_scale(X, feature_range=(0, 1)):
    lo, hi = feature_range
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    return lo + (X - X_min) / (X_max - X_min) * (hi - lo)

def robust_scale(X):
    """Scale by median and IQR (robust to outliers)."""
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return (X - median) / iqr

def l2_normalize(X):
    """Unit norm per sample."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)

# --- demo ---
np.random.seed(42)
X = np.column_stack([
    np.random.randn(100) * 1000 + 5000,   # large scale
    np.random.randn(100) * 0.01 + 0.5,    # tiny scale
    np.random.exponential(2, 100),          # skewed
])
# add an outlier
X[0] = [50000, 10, 100]

def stats(X, name):
    print(f"  {name:>15}: mean={X.mean(0).round(2)}, std={X.std(0).round(2)}, "
          f"min={X.min(0).round(2)}, max={X.max(0).round(2)}")

print("Feature Scaling Comparison:")
stats(X, "Original")
stats(standardize(X), "Standardized")
stats(min_max_scale(X), "Min-Max [0,1]")
stats(robust_scale(X), "Robust")
