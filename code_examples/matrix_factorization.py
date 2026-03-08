"""Matrix Factorization — NMF and ALS for recommendation."""

import numpy as np

def nmf(V, k, max_iter=200, tol=1e-4):
    """Non-negative Matrix Factorization: V ≈ W @ H."""
    m, n = V.shape
    W = np.random.rand(m, k) + 0.1
    H = np.random.rand(k, n) + 0.1
    for i in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-8)
        W *= (V @ H.T) / (W @ H @ H.T + 1e-8)
        if i % 50 == 0:
            err = np.linalg.norm(V - W @ H, 'fro')
            print(f"  Iter {i:3d}  error={err:.4f}")
    return W, H

def als(R, k, mask, max_iter=50, reg=0.1):
    """Alternating Least Squares for collaborative filtering.
    R: rating matrix, mask: 1 where rating exists."""
    m, n = R.shape
    U = np.random.randn(m, k) * 0.1
    V = np.random.randn(n, k) * 0.1
    for iteration in range(max_iter):
        # fix V, solve for U
        for i in range(m):
            idx = mask[i] > 0
            if idx.sum() == 0: continue
            Vi = V[idx]
            U[i] = np.linalg.solve(Vi.T @ Vi + reg * np.eye(k), Vi.T @ R[i, idx])
        # fix U, solve for V
        for j in range(n):
            idx = mask[:, j] > 0
            if idx.sum() == 0: continue
            Uj = U[idx]
            V[j] = np.linalg.solve(Uj.T @ Uj + reg * np.eye(k), Uj.T @ R[idx, j])
        pred = U @ V.T
        err = np.sqrt(np.mean((mask * (R - pred))**2))
        if iteration % 10 == 0:
            print(f"  Iter {iteration:3d}  RMSE={err:.4f}")
    return U, V

# --- demo ---
np.random.seed(42)

# NMF on a small non-negative matrix
print("=== Non-negative Matrix Factorization ===")
V = np.random.rand(6, 4) * 5
W, H = nmf(V, k=2, max_iter=200)
print(f"  Reconstruction error: {np.linalg.norm(V - W @ H, 'fro'):.4f}")

# ALS for recommendation
print("\n=== ALS Collaborative Filtering ===")
R = np.array([[5,3,0,1], [4,0,0,1], [1,1,0,5], [0,0,4,4], [0,1,5,4]], dtype=float)
mask = (R > 0).astype(float)
U, V = als(R, k=2, mask=mask)
pred = U @ V.T
print(f"\nPredicted ratings (0 = was missing):")
items = ["Action", "Comedy", "Drama", "SciFi"]
for i in range(5):
    for j in range(4):
        if R[i, j] == 0:
            print(f"  User {i} → {items[j]}: {pred[i,j]:.1f}")
