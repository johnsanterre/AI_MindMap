"""Eigenvalues — power iteration, applications in PCA and PageRank."""

import numpy as np

def power_iteration(A, n_iter=100):
    """Find dominant eigenvalue and eigenvector."""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = A @ v
        eigenvalue = v @ Av
        v = Av / np.linalg.norm(Av)
    return eigenvalue, v

def qr_algorithm(A, n_iter=100):
    """Find all eigenvalues via QR iteration."""
    Ak = A.copy().astype(float)
    for _ in range(n_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)

# --- demo ---
np.random.seed(42)

# symmetric matrix (guaranteed real eigenvalues)
M = np.random.randn(4, 4)
A = M + M.T  # make symmetric

# power iteration
val, vec = power_iteration(A)
true_vals, true_vecs = np.linalg.eigh(A)

print("=== Power Iteration ===")
print(f"  Dominant eigenvalue: {val:.4f} (true: {true_vals[-1]:.4f})")

# QR algorithm
qr_vals = qr_algorithm(A)
print(f"\n=== QR Algorithm ===")
print(f"  Computed: {np.sort(qr_vals).round(3)}")
print(f"  True:     {np.sort(true_vals).round(3)}")

# eigenvalue applications
print(f"\n=== Properties ===")
print(f"  Trace(A) = {np.trace(A):.3f}, Sum(eigenvalues) = {true_vals.sum():.3f}")
print(f"  Det(A)   = {np.linalg.det(A):.3f}, Product(eigenvalues) = {np.prod(true_vals):.3f}")
print(f"  Positive definite? {(true_vals > 0).all()}")
