"""K-Nearest Neighbors — classification and regression."""

import numpy as np
from collections import Counter

def knn_classify(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        nearest = y_train[dists.argsort()[:k]]
        preds.append(Counter(nearest).most_common(1)[0][0])
    return np.array(preds)

def knn_regress(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        nearest = y_train[dists.argsort()[:k]]
        preds.append(nearest.mean())
    return np.array(preds)

# --- demo ---
np.random.seed(42)
X_train = np.vstack([np.random.randn(30, 2) + [2, 2],
                     np.random.randn(30, 2) + [-2, -2]])
y_train = np.array([0]*30 + [1]*30)

X_test = np.array([[0, 0], [3, 3], [-3, -3], [1, -1]])
preds = knn_classify(X_train, y_train, X_test, k=5)

print("Classification (k=5):")
for x, p in zip(X_test, preds):
    print(f"  {x} → class {p}")

# regression demo
y_reg = X_train[:, 0] * 2 + 1 + np.random.randn(60) * 0.5
reg_preds = knn_regress(X_train, y_reg, X_test, k=5)
print("\nRegression (k=5):")
for x, p in zip(X_test, reg_preds):
    print(f"  {x} → {p:.2f}")
