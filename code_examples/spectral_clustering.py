"""Spectral Clustering — clustering via graph Laplacian eigenvectors."""

import numpy as np

def rbf_kernel(X, sigma=1.0):
    dists = np.sum((X[:, None] - X[None, :])**2, axis=2)
    return np.exp(-dists / (2 * sigma**2))

def graph_laplacian(W):
    D = np.diag(W.sum(axis=1))
    return D - W

def normalized_laplacian(W):
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
    L = graph_laplacian(W)
    return D_inv_sqrt @ L @ D_inv_sqrt

def spectral_cluster(X, k=2, sigma=1.0):
    W = rbf_kernel(X, sigma)
    np.fill_diagonal(W, 0)
    L_norm = normalized_laplacian(W)
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    # use k smallest eigenvectors (skip first ~0 eigenvalue)
    features = eigvecs[:, :k]
    # normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    features = features / norms
    # k-means on spectral features
    return simple_kmeans(features, k)

def simple_kmeans(X, k, n_iter=50):
    centers = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        dists = np.sum((X[:, None] - centers[None, :])**2, axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = X[labels == j].mean(axis=0)
    return labels

# --- demo ---
np.random.seed(42)

# two concentric rings (non-convex clusters)
n = 50
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
ring1 = np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(n, 2) * 0.1
ring2 = 3 * np.column_stack([np.cos(theta), np.sin(theta)]) + np.random.randn(n, 2) * 0.1
X = np.vstack([ring1, ring2])
true_labels = np.array([0]*n + [1]*n)

spec_labels = spectral_cluster(X, k=2, sigma=1.0)

# align labels
if (spec_labels != true_labels).mean() > 0.5:
    spec_labels = 1 - spec_labels

acc = (spec_labels == true_labels).mean()
print(f"=== Spectral Clustering on Concentric Rings ===")
print(f"Points: {len(X)}, True clusters: 2")
print(f"Accuracy: {acc:.1%}")
print(f"\nSpectral clustering handles non-convex shapes that k-means cannot.")
