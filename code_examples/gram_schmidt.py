"""Gram-Schmidt — orthogonalizing a set of vectors."""

import numpy as np

def gram_schmidt(V):
    """Classical Gram-Schmidt orthonormalization."""
    n, d = V.shape
    Q = np.zeros_like(V, dtype=float)
    for i in range(n):
        q = V[i].copy().astype(float)
        for j in range(i):
            q -= np.dot(Q[j], V[i]) * Q[j]
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            continue
        Q[i] = q / norm
    return Q

def modified_gram_schmidt(V):
    """Numerically stable modified Gram-Schmidt."""
    n, d = V.shape
    Q = V.copy().astype(float)
    for i in range(n):
        Q[i] /= np.linalg.norm(Q[i])
        for j in range(i+1, n):
            Q[j] -= np.dot(Q[i], Q[j]) * Q[i]
    return Q

# --- demo ---
np.random.seed(42)

print("=== Gram-Schmidt Orthogonalization ===\n")

# 3 vectors in R³
V = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
], dtype=float)

Q = gram_schmidt(V)

print("Original vectors:")
for i, v in enumerate(V):
    print(f"  v{i} = {v}")

print(f"\nOrthonormal basis:")
for i, q in enumerate(Q):
    print(f"  q{i} = {q.round(4)}")

print(f"\nVerification (Q @ Q^T should be I):")
print((Q @ Q.T).round(6))

# numerical stability comparison
print(f"\n--- Numerical Stability ---")
# near-parallel vectors (ill-conditioned)
eps = 1e-8
V_bad = np.array([[1, eps, 0], [1, 0, eps], [1, eps, eps]], dtype=float)

Q_classic = gram_schmidt(V_bad)
Q_modified = modified_gram_schmidt(V_bad)

ortho_err_classic = np.linalg.norm(Q_classic @ Q_classic.T - np.eye(3))
ortho_err_modified = np.linalg.norm(Q_modified @ Q_modified.T - np.eye(3))
print(f"  Classical GS orthogonality error: {ortho_err_classic:.2e}")
print(f"  Modified GS orthogonality error:  {ortho_err_modified:.2e}")

# QR decomposition connection
print(f"\n--- QR Decomposition ---")
A = np.random.randn(4, 3)
Q = modified_gram_schmidt(A.T).T
R = Q.T @ A
print(f"  A = Q @ R")
print(f"  Reconstruction error: {np.linalg.norm(A - Q @ R):.2e}")
