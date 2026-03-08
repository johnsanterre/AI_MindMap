"""Kernel PCA — nonlinear dimensionality reduction via the kernel trick."""

import numpy as np

def linear_kernel(X):
    return X @ X.T

def rbf_kernel(X, gamma=1.0):
    dists = np.sum((X[:, None] - X[None, :])**2, axis=2)
    return np.exp(-gamma * dists)

def polynomial_kernel(X, degree=3, c=1.0):
    return (X @ X.T + c)**degree

def kernel_pca(X, kernel_fn, n_components=2):
    K = kernel_fn(X)
    # center the kernel matrix
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n
    K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n

    eigvals, eigvecs = np.linalg.eigh(K_centered)
    idx = eigvals.argsort()[::-1][:n_components]
    return eigvecs[:, idx] * np.sqrt(np.abs(eigvals[idx]))

# --- demo ---
np.random.seed(42)

# concentric circles (nonlinear structure)
n = 100
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
inner = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(n, 2) * 0.1
outer = 3 * np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(n, 2) * 0.1
X = np.vstack([inner, outer])
labels = np.array([0]*n + [1]*n)

print("=== Kernel PCA ===")
print(f"Data: {len(X)} points, 2D concentric circles\n")

# linear PCA
from_linear = kernel_pca(X, linear_kernel, 1)
# RBF kernel PCA
from_rbf = kernel_pca(X, lambda X: rbf_kernel(X, gamma=0.5), 1)

# check separability
print("--- Separability in 1D projection ---")
for name, proj in [("Linear PCA", from_linear), ("RBF Kernel PCA", from_rbf)]:
    inner_proj = proj[labels == 0].flatten()
    outer_proj = proj[labels == 1].flatten()
    overlap = max(0, min(inner_proj.max(), outer_proj.max()) - max(inner_proj.min(), outer_proj.min()))
    inner_range = inner_proj.max() - inner_proj.min()
    outer_range = outer_proj.max() - outer_proj.min()
    separation = abs(inner_proj.mean() - outer_proj.mean()) / (inner_proj.std() + outer_proj.std())
    print(f"  {name}:")
    print(f"    Inner mean: {inner_proj.mean():>7.3f}, Outer mean: {outer_proj.mean():>7.3f}")
    print(f"    Separation ratio: {separation:.3f}")

print(f"\nRBF Kernel PCA separates circles that linear PCA cannot.")
