"""Pseudoinverse — solving over/underdetermined systems via Moore-Penrose."""

import numpy as np

def pseudoinverse_svd(A, tol=1e-10):
    """A⁺ = V Σ⁺ U^T via SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > tol, 1/s, 0)
    return (Vt.T * s_inv) @ U.T

def least_squares_solve(A, b):
    """Overdetermined: min ||Ax - b||² → x = A⁺ b."""
    return pseudoinverse_svd(A) @ b

def min_norm_solve(A, b):
    """Underdetermined: min ||x|| s.t. Ax = b → x = A⁺ b."""
    return pseudoinverse_svd(A) @ b

# --- demo ---
np.random.seed(42)

print("=== Moore-Penrose Pseudoinverse ===\n")

# overdetermined system (more equations than unknowns)
print("--- Overdetermined: 5 equations, 2 unknowns ---")
A_over = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
b_over = np.array([1.1, 2.9, 5.2, 6.8, 9.1])
x_over = least_squares_solve(A_over, b_over)
residual = np.linalg.norm(A_over @ x_over - b_over)
print(f"A⁺b = {x_over.round(4)}")
print(f"Residual: {residual:.4f} (best fit line: y = {x_over[0]:.2f} + {x_over[1]:.2f}x)")

# underdetermined system (more unknowns than equations)
print(f"\n--- Underdetermined: 2 equations, 4 unknowns ---")
A_under = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
b_under = np.array([10, 26], dtype=float)
x_under = min_norm_solve(A_under, b_under)
print(f"Min-norm solution: {x_under.round(4)}")
print(f"||x|| = {np.linalg.norm(x_under):.4f}")
print(f"Ax = {(A_under @ x_under).round(4)} (should be {b_under})")

# verify properties
print(f"\n--- Pseudoinverse Properties ---")
A = np.random.randn(4, 3)
A_pinv = pseudoinverse_svd(A)
print(f"A: {A.shape}, A⁺: {A_pinv.shape}")
print(f"A A⁺ A = A:   {np.allclose(A @ A_pinv @ A, A)}")
print(f"A⁺ A A⁺ = A⁺: {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")
print(f"(A A⁺)^T = A A⁺: {np.allclose((A @ A_pinv).T, A @ A_pinv)}")

# comparison with numpy
A_pinv_np = np.linalg.pinv(A)
print(f"\nMatches numpy pinv: {np.allclose(A_pinv, A_pinv_np)}")
