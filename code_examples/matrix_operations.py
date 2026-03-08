"""Matrix Operations — fundamental linear algebra operations."""

import numpy as np

def row_echelon(A):
    """Gaussian elimination to row echelon form."""
    M = A.astype(float).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # find pivot
        pivot = None
        for i in range(r, rows):
            if abs(M[i, c]) > 1e-10:
                pivot = i
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        M[r] /= M[r, c]
        for i in range(r+1, rows):
            M[i] -= M[i, c] * M[r]
        r += 1
    return M

def gram_schmidt(V):
    """Orthogonalize columns of V."""
    n, m = V.shape
    U = np.zeros_like(V, dtype=float)
    for i in range(m):
        u = V[:, i].astype(float)
        for j in range(i):
            u -= np.dot(U[:, j], V[:, i]) * U[:, j]
        U[:, i] = u / np.linalg.norm(u)
    return U

def matrix_rank(A, tol=1e-10):
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))

# --- demo ---
np.random.seed(42)

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)

print("=== Matrix A ===")
print(A)
print(f"\n  Determinant: {np.linalg.det(A):.1f}")
print(f"  Rank: {matrix_rank(A)}")
print(f"  Trace: {np.trace(A):.1f}")

print(f"\n=== Row Echelon Form ===")
print(row_echelon(A).round(3))

print(f"\n=== Inverse ===")
print(np.linalg.inv(A).round(3))

print(f"\n=== Gram-Schmidt ===")
V = np.random.randn(3, 3)
Q = gram_schmidt(V)
print("Orthogonality check (Q^T Q):")
print((Q.T @ Q).round(6))
