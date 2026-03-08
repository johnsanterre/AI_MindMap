"""QR Decomposition — factoring matrices for least squares and eigenvalues."""

import numpy as np

def qr_householder(A):
    """QR decomposition using Householder reflections."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for j in range(min(m-1, n)):
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        v = x + e1
        v = v / (np.linalg.norm(v) + 1e-15)

        H_sub = np.eye(len(x)) - 2 * np.outer(v, v)
        H = np.eye(m)
        H[j:, j:] = H_sub

        R = H @ R
        Q = Q @ H.T

    return Q, R

def qr_solve(A, b):
    """Solve Ax = b using QR decomposition."""
    Q, R = qr_householder(A)
    # x = R⁻¹ Q^T b (back substitution)
    Qtb = Q.T @ b
    n = A.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Qtb[i] - R[i, i+1:n] @ x[i+1:n]) / R[i, i]
    return x

def qr_eigenvalues(A, n_iter=100):
    """QR algorithm for eigenvalues."""
    Ak = A.copy().astype(float)
    for _ in range(n_iter):
        Q, R = qr_householder(Ak)
        Ak = R @ Q
    return np.diag(Ak)

# --- demo ---
np.random.seed(42)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [1, 0, 1]], dtype=float)
Q, R = qr_householder(A)

print("=== QR Decomposition ===")
print(f"A ({A.shape[0]}×{A.shape[1]}):\n{A}\n")
print(f"Q (orthogonal, {Q.shape[0]}×{Q.shape[1]}):")
print(Q.round(4))
print(f"\nR (upper triangular, {R.shape[0]}×{R.shape[1]}):")
print(R.round(4))
print(f"\n||A - QR|| = {np.linalg.norm(A - Q @ R):.2e}")
print(f"||Q^TQ - I|| = {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.2e}")

# least squares via QR
print(f"\n--- Least Squares via QR ---")
b = np.array([1, 2, 3, 4], dtype=float)
x_qr = qr_solve(A, b)
x_np = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"QR solution: {x_qr.round(6)}")
print(f"NumPy lstsq: {x_np.round(6)}")

# eigenvalue computation
print(f"\n--- Eigenvalues via QR Algorithm ---")
M = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
eigs_qr = np.sort(qr_eigenvalues(M))
eigs_np = np.sort(np.linalg.eigvalsh(M))
print(f"QR algorithm: {eigs_qr.round(4)}")
print(f"NumPy:        {eigs_np.round(4)}")
