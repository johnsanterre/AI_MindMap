"""Determinant — computing and interpreting matrix determinants."""

import numpy as np

def det_2x2(A):
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]

def det_cofactor(A):
    """Determinant via cofactor expansion (recursive)."""
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    if n == 2:
        return det_2x2(A)
    det = 0
    for j in range(n):
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        det += (-1)**j * A[0, j] * det_cofactor(minor)
    return det

def det_lu(A):
    """Determinant via row reduction (O(n³))."""
    n = A.shape[0]
    U = A.copy().astype(float)
    sign = 1
    for j in range(n-1):
        # partial pivoting
        pivot = np.argmax(np.abs(U[j:, j])) + j
        if pivot != j:
            U[[j, pivot]] = U[[pivot, j]]
            sign *= -1
        if abs(U[j, j]) < 1e-12:
            return 0.0
        for i in range(j+1, n):
            U[i, j:] -= (U[i, j] / U[j, j]) * U[j, j:]
    return sign * np.prod(np.diag(U))

# --- demo ---
np.random.seed(42)

print("=== Determinant ===\n")

# geometric interpretation: area/volume scaling
A = np.array([[2, 1], [0, 3]], dtype=float)
print(f"A = [[2,1],[0,3]]")
print(f"det(A) = {det_2x2(A):.0f}")
print(f"A scales area by factor {abs(det_2x2(A)):.0f}\n")

# properties
I = np.eye(3)
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)
print("--- Properties ---")
print(f"det(I) = {det_cofactor(I):.0f}")
print(f"det(B·C) = {det_lu(B @ C):.4f}")
print(f"det(B)·det(C) = {det_lu(B) * det_lu(C):.4f}")
print(f"det(B^T) = det(B): {np.isclose(det_lu(B.T), det_lu(B))}")
print(f"det(2B) = 2³·det(B): {np.isclose(det_lu(2*B), 8*det_lu(B))}")

# singular matrix
print(f"\n--- Singularity Check ---")
S = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
print(f"Singular matrix det = {det_lu(S):.6f} (≈ 0)")

# comparison of methods
print(f"\n--- Method Comparison ---")
for n in [3, 4, 5]:
    M = np.random.randn(n, n)
    d_cof = det_cofactor(M)
    d_lu = det_lu(M)
    d_np = np.linalg.det(M)
    print(f"  {n}×{n}: cofactor={d_cof:.4f}, LU={d_lu:.4f}, numpy={d_np:.4f}")
