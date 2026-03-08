"""Cholesky Decomposition — efficient factorization for positive definite matrices."""

import numpy as np

def cholesky(A):
    """A = L @ L.T where L is lower triangular."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky: A = LL^T → Ly = b, L^Tx = y."""
    L = cholesky(A)
    n = len(b)
    # forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    # back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L[i, i]
    return x

def sample_multivariate_normal(mean, cov, n_samples):
    """Sample from N(mean, cov) using Cholesky."""
    L = cholesky(cov)
    z = np.random.randn(n_samples, len(mean))
    return mean + z @ L.T

# --- demo ---
np.random.seed(42)

# SPD matrix
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
L = cholesky(A)

print("=== Cholesky Decomposition ===")
print(f"A = L @ L^T\n")
print(f"A:\n{A}\n")
print(f"L (lower triangular):\n{L.round(4)}\n")
print(f"||A - LL^T|| = {np.linalg.norm(A - L @ L.T):.2e}")

# solve linear system
b = np.array([1, 2, 3], dtype=float)
x = cholesky_solve(A, b)
print(f"\nSolving Ax = b:")
print(f"  x = {x.round(6)}")
print(f"  ||Ax - b|| = {np.linalg.norm(A @ x - b):.2e}")

# multivariate normal sampling
print(f"\n--- Sampling Multivariate Normal ---")
mean = np.array([1.0, 2.0, 3.0])
cov = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
samples = sample_multivariate_normal(mean, cov, 10000)
print(f"Target mean: {mean}")
print(f"Sample mean: {samples.mean(axis=0).round(3)}")
print(f"Target cov diagonal: {np.diag(cov)}")
print(f"Sample cov diagonal: {np.diag(np.cov(samples.T)).round(3)}")

# speed comparison note
print(f"\nCholesky is 2× faster than LU for SPD systems.")
print(f"Used in: GP regression, Kalman filters, sampling.")
