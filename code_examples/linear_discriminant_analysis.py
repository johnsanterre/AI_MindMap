"""Linear Discriminant Analysis — dimensionality reduction for classification."""

import numpy as np

def lda(X, y, n_components=1):
    classes = np.unique(y)
    n_features = X.shape[1]
    mean_overall = X.mean(axis=0)

    S_W = np.zeros((n_features, n_features))  # within-class scatter
    S_B = np.zeros((n_features, n_features))  # between-class scatter

    for c in classes:
        X_c = X[y == c]
        mean_c = X_c.mean(axis=0)
        S_W += (X_c - mean_c).T @ (X_c - mean_c)
        n_c = len(X_c)
        diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * (diff @ diff.T)

    # solve generalized eigenvalue problem
    A = np.linalg.inv(S_W + 1e-6 * np.eye(n_features)) @ S_B
    eigvals, eigvecs = np.linalg.eigh(A)
    # sort descending
    idx = eigvals.argsort()[::-1]
    W = eigvecs[:, idx[:n_components]]
    return W, eigvals[idx[:n_components]]

# --- demo ---
np.random.seed(42)

# 3 classes in 4D
n = 50
X0 = np.random.randn(n, 4) + [2, 0, 0, 0]
X1 = np.random.randn(n, 4) + [0, 2, 0, 0]
X2 = np.random.randn(n, 4) + [0, 0, 2, 0]
X = np.vstack([X0, X1, X2])
y = np.array([0]*n + [1]*n + [2]*n)

W, eigvals = lda(X, y, n_components=2)
X_proj = X @ W

print("=== Linear Discriminant Analysis ===")
print(f"Original: {X.shape[1]}D → Projected: {W.shape[1]}D\n")
print(f"Discriminant eigenvalues: {eigvals.round(3)}")
print(f"\nClass means in LDA space:")
for c in range(3):
    mean = X_proj[y == c].mean(axis=0)
    print(f"  Class {c}: ({mean[0]:.3f}, {mean[1]:.3f})")

# simple classifier in LDA space
from collections import Counter
correct = 0
for i in range(len(X)):
    dists = [np.linalg.norm(X_proj[i] - X_proj[y == c].mean(axis=0)) for c in range(3)]
    if np.argmin(dists) == y[i]:
        correct += 1
print(f"\nNearest-centroid accuracy in LDA space: {correct/len(X):.1%}")
