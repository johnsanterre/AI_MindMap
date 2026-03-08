"""PCA — Principal Component Analysis from scratch."""

import numpy as np

def pca(X, n_components):
    # center data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # covariance matrix
    cov = (X_centered.T @ X_centered) / (len(X) - 1)

    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # sort by descending eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # project
    components = eigenvectors[:, :n_components]
    X_projected = X_centered @ components

    explained_var = eigenvalues[:n_components] / eigenvalues.sum()
    return X_projected, components, explained_var

# --- demo ---
np.random.seed(42)
# 5D data with 2 real dimensions
t = np.random.randn(100)
s = np.random.randn(100)
X = np.column_stack([
    2*t + 0.1*np.random.randn(100),
    -t + s + 0.1*np.random.randn(100),
    t + 2*s + 0.1*np.random.randn(100),
    3*t - s + 0.1*np.random.randn(100),
    -2*t + 3*s + 0.1*np.random.randn(100),
])

X_2d, components, var_explained = pca(X, 2)
print(f"Original shape: {X.shape}")
print(f"Reduced shape:  {X_2d.shape}")
print(f"Variance explained: {var_explained.round(3)}")
print(f"Total explained:    {var_explained.sum():.1%}")
print(f"\nFirst 5 projected points:\n{X_2d[:5].round(3)}")
