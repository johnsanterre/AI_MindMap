"""Spectral Normalization — stabilizing GAN training by constraining Lipschitz constant."""

import numpy as np

def spectral_norm(W, n_iter=10):
    """Estimate largest singular value via power iteration."""
    u = np.random.randn(W.shape[0])
    u /= np.linalg.norm(u)
    for _ in range(n_iter):
        v = W.T @ u
        v /= np.linalg.norm(v) + 1e-8
        u = W @ v
        u /= np.linalg.norm(u) + 1e-8
    sigma = u @ W @ v
    return sigma

def normalize_weight(W, n_iter=10):
    """W_SN = W / σ(W)."""
    sigma = spectral_norm(W, n_iter)
    return W / sigma

# --- demo ---
np.random.seed(42)

print("=== Spectral Normalization ===\n")

# random weight matrix
W = np.random.randn(8, 8) * 2

sigma_est = spectral_norm(W)
sigma_exact = np.linalg.svd(W, compute_uv=False)[0]
print(f"Weight matrix: {W.shape}")
print(f"Spectral norm (power iter): {sigma_est:.4f}")
print(f"Spectral norm (exact SVD):  {sigma_exact:.4f}")

W_norm = normalize_weight(W)
sigma_after = spectral_norm(W_norm)
print(f"After normalization: σ = {sigma_after:.4f} (should be ≈ 1)")

# Lipschitz bound demonstration
print(f"\n--- Lipschitz Bound ---")
n_tests = 100
max_ratio = 0
for _ in range(n_tests):
    x1 = np.random.randn(8)
    x2 = np.random.randn(8)
    out_diff = np.linalg.norm(W_norm @ x1 - W_norm @ x2)
    in_diff = np.linalg.norm(x1 - x2)
    ratio = out_diff / (in_diff + 1e-8)
    max_ratio = max(max_ratio, ratio)

print(f"  Max ||f(x1)-f(x2)|| / ||x1-x2|| = {max_ratio:.4f}")
print(f"  Lipschitz constant bounded by σ = {sigma_after:.4f}")

# effect on layer outputs
print(f"\n--- Output scale comparison ---")
x = np.random.randn(100, 8)
out_raw = x @ W.T
out_sn = x @ W_norm.T
print(f"  Raw layer output std:        {out_raw.std():.3f}")
print(f"  Spectral norm output std:    {out_sn.std():.3f}")
print(f"  SN prevents output explosion in deep networks (esp. GANs).")
