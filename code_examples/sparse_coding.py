"""Sparse Coding — learning overcomplete dictionaries with sparse activations."""

import numpy as np

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ista(D, y, lam=0.1, n_iter=100):
    """Iterative Shrinkage-Thresholding for sparse coding."""
    L = np.linalg.norm(D.T @ D, ord=2)  # Lipschitz constant
    x = np.zeros(D.shape[1])
    for _ in range(n_iter):
        grad = D.T @ (D @ x - y)
        x = soft_threshold(x - grad / L, lam / L)
    return x

def learn_dictionary(Y, n_atoms, lam=0.1, n_iter=50):
    """Alternating minimization: sparse code then update dictionary."""
    n_samples, d = Y.shape
    D = np.random.randn(d, n_atoms)
    D /= np.linalg.norm(D, axis=0, keepdims=True)

    for it in range(n_iter):
        # sparse coding step
        X = np.array([ista(D, y, lam) for y in Y])
        # dictionary update step
        for j in range(n_atoms):
            if np.all(X[:, j] == 0):
                continue
            residual = Y - X @ D.T + np.outer(X[:, j], D[:, j])
            D[:, j] = residual.T @ X[:, j]
            norm = np.linalg.norm(D[:, j])
            if norm > 0:
                D[:, j] /= norm
    return D, X

# --- demo ---
np.random.seed(42)

# generate signals from sparse combinations
d = 8
n_atoms = 12  # overcomplete
D_true = np.random.randn(d, n_atoms)
D_true /= np.linalg.norm(D_true, axis=0, keepdims=True)

n_samples = 50
X_true = np.zeros((n_samples, n_atoms))
for i in range(n_samples):
    active = np.random.choice(n_atoms, 2, replace=False)
    X_true[i, active] = np.random.randn(2)
Y = X_true @ D_true.T + np.random.randn(n_samples, d) * 0.01

D_learned, X_learned = learn_dictionary(Y, n_atoms, lam=0.1, n_iter=30)

# reconstruction
Y_recon = X_learned @ D_learned.T
recon_error = np.mean((Y - Y_recon)**2)
avg_sparsity = np.mean(np.abs(X_learned) > 0.01)

print("=== Sparse Coding ===")
print(f"Signal dim: {d}, Dictionary atoms: {n_atoms} (overcomplete)")
print(f"Samples: {n_samples}\n")
print(f"Reconstruction MSE: {recon_error:.6f}")
print(f"Avg active coefficients: {avg_sparsity * n_atoms:.1f} / {n_atoms}")
print(f"True sparsity: 2 / {n_atoms}")
