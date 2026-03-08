"""Low-Rank Approximation — compressing matrices via truncated SVD."""

import numpy as np

def truncated_svd(A, k):
    """Best rank-k approximation (Eckart-Young theorem)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]

def reconstruct(U, s, Vt):
    return U * s @ Vt

def compression_ratio(shape, k):
    m, n = shape
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original

# --- demo ---
np.random.seed(42)

# image-like matrix (smooth, low intrinsic rank)
t = np.linspace(0, 2*np.pi, 50)
A = np.outer(np.sin(t), np.cos(t)) + \
    0.5 * np.outer(np.sin(2*t), np.cos(3*t)) + \
    np.random.randn(50, 50) * 0.1

print("=== Low-Rank Approximation ===")
print(f"Matrix: {A.shape[0]}×{A.shape[1]}\n")

# singular value spectrum
U_full, s_full, Vt_full = np.linalg.svd(A)
energy = np.cumsum(s_full**2) / np.sum(s_full**2)
print("Singular value spectrum:")
print(f"{'k':>4} {'σ_k':>10} {'Energy %':>10} {'Compression':>12}")
print("-" * 40)
for k in [1, 2, 3, 5, 10, 20]:
    U_k, s_k, Vt_k = truncated_svd(A, k)
    A_k = reconstruct(U_k, s_k, Vt_k)
    error = np.linalg.norm(A - A_k) / np.linalg.norm(A)
    cr = compression_ratio(A.shape, k)
    print(f"{k:>4} {s_full[k-1]:>10.3f} {energy[k-1]*100:>9.1f}% {cr:>11.1%}")

# text matrix example
print(f"\n--- Document-Term Matrix ---")
# 100 docs, 500 terms, but only ~5 topics
n_docs, n_terms, n_topics = 100, 500, 5
W = np.random.rand(n_docs, n_topics)
H = np.random.rand(n_topics, n_terms)
M = W @ H + np.random.randn(n_docs, n_terms) * 0.01

for k in [2, 5, 10]:
    U, s, Vt = truncated_svd(M, k)
    err = np.linalg.norm(M - reconstruct(U, s, Vt)) / np.linalg.norm(M)
    cr = compression_ratio(M.shape, k)
    print(f"  rank-{k}: error={err:.4f}, storage={cr:.1%}")
