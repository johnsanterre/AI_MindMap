"""SVD — Singular Value Decomposition for compression and recommendation."""

import numpy as np

def low_rank_approx(A, k):
    """Rank-k approximation via truncated SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def compression_ratio(shape, k):
    m, n = shape
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original

def collaborative_filter(ratings, k=2):
    """Simple SVD-based collaborative filtering."""
    # fill missing (0) with row means
    filled = ratings.copy()
    for i in range(filled.shape[0]):
        mask = filled[i] > 0
        if mask.any():
            filled[i, ~mask] = filled[i, mask].mean()
    approx = low_rank_approx(filled, k)
    return approx

# --- demo ---
np.random.seed(42)

# 1) Matrix compression
print("=== Low-Rank Approximation ===")
A = np.random.randn(20, 10)
for k in [1, 2, 5, 10]:
    Ak = low_rank_approx(A, k)
    error = np.linalg.norm(A - Ak, 'fro') / np.linalg.norm(A, 'fro')
    ratio = compression_ratio(A.shape, k)
    print(f"  Rank {k:2d}: relative error={error:.4f}, compression={ratio:.1%}")

# 2) Collaborative filtering
print("\n=== Collaborative Filtering ===")
# users x items (0 = unseen)
ratings = np.array([
    [5, 4, 0, 0, 1],
    [4, 0, 3, 0, 1],
    [0, 0, 5, 4, 0],
    [0, 3, 4, 5, 0],
    [1, 1, 0, 0, 5],
])
users = ["Alice", "Bob", "Carol", "Dave", "Eve"]
items = ["Action", "Comedy", "Drama", "Romance", "Horror"]

predicted = collaborative_filter(ratings, k=2)
print("Predicted ratings (0 = was missing):")
for i, user in enumerate(users):
    for j, item in enumerate(items):
        if ratings[i, j] == 0:
            print(f"  {user} → {item}: {predicted[i, j]:.1f} (predicted)")
