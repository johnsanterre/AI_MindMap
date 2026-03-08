"""Conjugate Gradient — efficient solver for symmetric positive-definite systems."""

import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=100):
    """Solve Ax = b using conjugate gradient method."""
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    r = b - A @ x
    p = r.copy()
    history = [np.linalg.norm(r)]

    for _ in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        history.append(np.linalg.norm(r_new))
        if np.linalg.norm(r_new) < tol:
            break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return x, history

def preconditioned_cg(A, b, M_inv, x0=None, tol=1e-10, max_iter=100):
    """Preconditioned CG: M⁻¹ approximates A⁻¹."""
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)
    r = b - A @ x
    z = M_inv @ r
    p = z.copy()
    history = [np.linalg.norm(r)]

    for _ in range(max_iter):
        Ap = A @ p
        alpha = (r @ z) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        history.append(np.linalg.norm(r_new))
        if np.linalg.norm(r_new) < tol:
            break
        z_new = M_inv @ r_new
        beta = (r_new @ z_new) / (r @ z)
        p = z_new + beta * p
        r, z = r_new, z_new

    return x, history

# --- demo ---
np.random.seed(42)

# SPD matrix
n = 50
R = np.random.randn(n, n)
A = R.T @ R + np.eye(n)  # guaranteed SPD
b = np.random.randn(n)

x_exact = np.linalg.solve(A, b)

x_cg, hist_cg = conjugate_gradient(A, b)
# Jacobi preconditioner: M = diag(A)
M_inv = np.diag(1.0 / np.diag(A))
x_pcg, hist_pcg = preconditioned_cg(A, b, M_inv)

print("=== Conjugate Gradient Method ===")
print(f"System size: {n}×{n}\n")
print(f"{'Iter':>5} {'CG Residual':>14} {'PCG Residual':>14}")
print("-" * 36)
for i in [0, 5, 10, 20, 30, min(len(hist_cg)-1, 50)]:
    cg = hist_cg[i] if i < len(hist_cg) else hist_cg[-1]
    pcg = hist_pcg[i] if i < len(hist_pcg) else hist_pcg[-1]
    print(f"{i:>5} {cg:>14.2e} {pcg:>14.2e}")

print(f"\nCG converged in {len(hist_cg)-1} iters")
print(f"PCG converged in {len(hist_pcg)-1} iters")
print(f"Solution error: {np.linalg.norm(x_cg - x_exact):.2e}")
print(f"Condition number: {np.linalg.cond(A):.1f}")
