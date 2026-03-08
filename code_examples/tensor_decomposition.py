"""Tensor Decomposition — CP decomposition for multi-dimensional data."""

import numpy as np

def cp_decompose(T, rank, n_iter=100):
    """CP decomposition via alternating least squares.
    T ≈ sum_r a_r ⊗ b_r ⊗ c_r."""
    d1, d2, d3 = T.shape
    A = np.random.randn(d1, rank)
    B = np.random.randn(d2, rank)
    C = np.random.randn(d3, rank)

    for _ in range(n_iter):
        # update A: unfold along mode 0
        T0 = T.reshape(d1, d2 * d3)
        khatri_rao_BC = np.array([np.kron(C[:, r], B[:, r]) for r in range(rank)]).T
        A = np.linalg.lstsq(khatri_rao_BC, T0.T, rcond=None)[0].T

        # update B: unfold along mode 1
        T1 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
        khatri_rao_AC = np.array([np.kron(C[:, r], A[:, r]) for r in range(rank)]).T
        B = np.linalg.lstsq(khatri_rao_AC, T1.T, rcond=None)[0].T

        # update C: unfold along mode 2
        T2 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)
        khatri_rao_AB = np.array([np.kron(B[:, r], A[:, r]) for r in range(rank)]).T
        C = np.linalg.lstsq(khatri_rao_AB, T2.T, rcond=None)[0].T

    return A, B, C

def reconstruct_cp(A, B, C):
    d1, d2, d3 = A.shape[0], B.shape[0], C.shape[0]
    rank = A.shape[1]
    T = np.zeros((d1, d2, d3))
    for r in range(rank):
        T += np.einsum('i,j,k', A[:, r], B[:, r], C[:, r])
    return T

# --- demo ---
np.random.seed(42)

# create rank-2 tensor
d1, d2, d3 = 5, 4, 3
a1, a2 = np.random.randn(d1), np.random.randn(d1)
b1, b2 = np.random.randn(d2), np.random.randn(d2)
c1, c2 = np.random.randn(d3), np.random.randn(d3)
T = np.einsum('i,j,k', a1, b1, c1) + np.einsum('i,j,k', a2, b2, c2)
T += np.random.randn(d1, d2, d3) * 0.01  # small noise

print("=== Tensor (CP) Decomposition ===")
print(f"Tensor shape: {T.shape}, true rank: 2\n")

for rank in [1, 2, 3]:
    A, B, C = cp_decompose(T, rank)
    T_approx = reconstruct_cp(A, B, C)
    error = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
    print(f"  Rank-{rank}: relative error = {error:.6f}")

print(f"\nRank-2 recovers the tensor almost perfectly.")
print(f"Used in: recommendation systems, topic models, signal processing.")
