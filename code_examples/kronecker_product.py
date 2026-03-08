"""Kronecker Product — block matrix operations for structured computation."""

import numpy as np

def kronecker(A, B):
    """A ⊗ B: each element a_ij replaced by a_ij * B."""
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m*p, n*q))
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
    return result

def kron_vec_product(A, B, x):
    """Efficient (A ⊗ B)x without forming the full Kronecker product.
    (A ⊗ B) vec(X) = vec(B X A^T) where x = vec(X)."""
    p, q = B.shape[0], A.shape[1]
    X = x.reshape(q, p).T  # careful with reshape order
    return (B @ X @ A.T).T.flatten()

# --- demo ---
np.random.seed(42)

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

K = kronecker(A, B)

print("=== Kronecker Product ===\n")
print(f"A (2×2):\n{A}\n")
print(f"B (2×2):\n{B}\n")
print(f"A ⊗ B (4×4):\n{K}\n")

# verify against numpy
K_np = np.kron(A, B)
print(f"Matches numpy: {np.allclose(K, K_np)}")

# properties
print(f"\n--- Properties ---")
print(f"(A⊗B)^T = A^T ⊗ B^T: {np.allclose(K.T, np.kron(A.T, B.T))}")
det_K = np.linalg.det(K)
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
print(f"det(A⊗B) = det(A)^q · det(B)^p: {det_K:.1f} = {det_A**2 * det_B**2:.1f}")

# efficient computation
print(f"\n--- Computational Efficiency ---")
m, n = 50, 50
A_big = np.random.randn(m, m)
B_big = np.random.randn(n, n)
x = np.random.randn(m * n)

# naive: form (m*n) × (m*n) matrix
# efficient: use B @ X @ A.T
K_full = np.kron(A_big, B_big)
result_naive = K_full @ x

print(f"Full Kronecker: {K_full.shape} = {K_full.size} elements")
print(f"Factors: A({m}×{m}) + B({n}×{n}) = {A_big.size + B_big.size} elements")
print(f"Memory ratio: {K_full.size / (A_big.size + B_big.size):.0f}×")
print(f"\nUsed in: multi-task learning, structured matrices, quantum computing.")
