"""Kernel Methods — kernel trick, kernel PCA, kernel regression."""

import numpy as np

def linear_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1.0):
    sq1 = np.sum(X1**2, axis=1, keepdims=True)
    sq2 = np.sum(X2**2, axis=1, keepdims=True)
    return np.exp(-gamma * (sq1 + sq2.T - 2 * X1 @ X2.T))

def polynomial_kernel(X1, X2, degree=3, c=1):
    return (X1 @ X2.T + c) ** degree

def kernel_ridge(X, y, kernel_fn, alpha=0.1):
    K = kernel_fn(X, X)
    w = np.linalg.solve(K + alpha * np.eye(len(X)), y)
    return w

def kernel_predict(X_train, X_test, w, kernel_fn):
    K = kernel_fn(X_test, X_train)
    return K @ w

def kernel_pca(X, kernel_fn, n_components=2):
    K = kernel_fn(X, X)
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    idx = eigenvalues.argsort()[::-1][:n_components]
    return eigenvectors[:, idx] * np.sqrt(np.abs(eigenvalues[idx]))

# --- demo ---
np.random.seed(42)

# nonlinear regression
X = np.sort(np.random.uniform(-3, 3, 50)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(50) * 0.1

w = kernel_ridge(X, y, lambda a, b: rbf_kernel(a, b, gamma=1.0), alpha=0.01)
X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
preds = kernel_predict(X, X_test, w, lambda a, b: rbf_kernel(a, b, gamma=1.0))

print("Kernel Ridge Regression (RBF):")
for x, p, t in zip(X_test.ravel(), preds, np.sin(X_test.ravel())):
    print(f"  x={x:+.1f}  pred={p:+.3f}  true={t:+.3f}")

# kernel PCA on circular data
theta = np.random.uniform(0, 2*np.pi, 100)
r = np.where(theta < np.pi, 1, 2) + np.random.randn(100) * 0.1
X_circle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
projected = kernel_pca(X_circle, lambda a, b: rbf_kernel(a, b, gamma=1.0), n_components=2)
print(f"\nKernel PCA: {X_circle.shape} → {projected.shape}")
