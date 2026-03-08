"""Power Iteration — finding dominant eigenvalue and eigenvector."""

import numpy as np

def power_iteration(A, n_iter=100, tol=1e-10):
    """Find largest eigenvalue and corresponding eigenvector."""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        Av = A @ v
        eigenvalue = v @ Av
        v_new = Av / np.linalg.norm(Av)
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return eigenvalue, v

def inverse_iteration(A, mu, n_iter=100, tol=1e-10):
    """Find eigenvalue closest to mu."""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    I = np.eye(n)

    eigenvalue = v @ A @ v
    for _ in range(n_iter):
        try:
            w = np.linalg.solve(A - mu * I, v)
        except np.linalg.LinAlgError:
            break
        v_new = w / np.linalg.norm(w)
        eigenvalue = v_new @ A @ v_new
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return eigenvalue, v

def deflation(A, eigenvalue, eigenvector):
    """Remove contribution of known eigenpair."""
    return A - eigenvalue * np.outer(eigenvector, eigenvector)

# --- demo ---
np.random.seed(42)

A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)

print("=== Power Iteration ===")
print(f"A:\n{A}\n")

eig1, v1 = power_iteration(A)
print(f"Largest eigenvalue:  {eig1:.6f}")
print(f"Eigenvector:         {v1.round(4)}")

# find all eigenvalues by deflation
A2 = deflation(A, eig1, v1)
eig2, v2 = power_iteration(A2)
A3 = deflation(A2, eig2, v2)
eig3, v3 = power_iteration(A3)

print(f"\nAll eigenvalues (power + deflation): {sorted([eig1, eig2, eig3], reverse=True)}")
print(f"NumPy eigenvalues:                   {sorted(np.linalg.eigvalsh(A), reverse=True)}")

# inverse iteration: find eigenvalue near target
print(f"\n--- Inverse Iteration ---")
for mu in [1.5, 3.0, 4.5]:
    eig, v = inverse_iteration(A, mu)
    print(f"  Target μ={mu:.1f} → eigenvalue={eig:.6f}")

# convergence rate
print(f"\nConvergence rate ∝ |λ₂/λ₁| = {abs(eig2/eig1):.3f}")
print(f"Smaller ratio → faster convergence.")
