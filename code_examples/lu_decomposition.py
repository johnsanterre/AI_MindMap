"""LU Decomposition — factoring matrices for solving linear systems."""

import numpy as np

def lu_decompose(A):
    """PA = LU decomposition with partial pivoting."""
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)

    for j in range(n-1):
        # partial pivoting
        pivot = np.argmax(np.abs(U[j:, j])) + j
        if pivot != j:
            U[[j, pivot]] = U[[pivot, j]]
            P[[j, pivot]] = P[[pivot, j]]
            if j > 0:
                L[[j, pivot], :j] = L[[pivot, j], :j]

        for i in range(j+1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]

    return P, L, U

def lu_solve(P, L, U, b):
    """Solve Ax = b given PA = LU."""
    n = len(b)
    Pb = P @ b
    # forward: Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]
    # backward: Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def determinant_lu(L, U, P):
    """det(A) = det(P)^{-1} * det(L) * det(U)."""
    sign = np.linalg.det(P)  # ±1
    return sign * np.prod(np.diag(U))

# --- demo ---
np.random.seed(42)

A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
P, L, U = lu_decompose(A)

print("=== LU Decomposition ===")
print(f"PA = LU\n")
print(f"A:\n{A}\nP:\n{P}\nL:\n{L.round(4)}\nU:\n{U.round(4)}")
print(f"\n||PA - LU|| = {np.linalg.norm(P @ A - L @ U):.2e}")

# solve multiple right-hand sides efficiently
print(f"\n--- Solving multiple systems ---")
for i, b in enumerate([[1,0,0], [0,1,0], [0,0,1]]):
    b = np.array(b, dtype=float)
    x = lu_solve(P, L, U, b)
    print(f"  b={b} → x={x.round(4)}")

# determinant
det = determinant_lu(L, U, P)
print(f"\ndet(A) = {det:.1f} (numpy: {np.linalg.det(A):.1f})")
print(f"LU: O(n³) once, then O(n²) per solve. Efficient for multiple right-hand sides.")
