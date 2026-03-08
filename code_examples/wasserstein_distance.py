"""Wasserstein Distance — Earth Mover's Distance between distributions."""

import numpy as np

def wasserstein_1d(p, q):
    """W1 distance between two 1D distributions (sorted samples)."""
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)
    # if different lengths, interpolate
    n = max(len(p), len(q))
    p_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(p)), p_sorted)
    q_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(q)), q_sorted)
    return np.mean(np.abs(p_interp - q_interp))

def wasserstein_cdf(p, q, n_points=1000):
    """W1 via CDF: integral of |F_p - F_q|."""
    all_vals = np.sort(np.concatenate([p, q]))
    lo, hi = all_vals[0], all_vals[-1]
    x = np.linspace(lo, hi, n_points)
    cdf_p = np.array([np.mean(p <= xi) for xi in x])
    cdf_q = np.array([np.mean(q <= xi) for xi in x])
    return np.trapezoid(np.abs(cdf_p - cdf_q), x)

def sinkhorn_distance(C, a, b, reg=0.1, n_iter=100):
    """Approximate OT distance via Sinkhorn algorithm."""
    n, m = C.shape
    K = np.exp(-C / reg)
    u = np.ones(n)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-10)
        u = a / (K @ v + 1e-10)
    T = np.diag(u) @ K @ np.diag(v)
    return np.sum(T * C)

# --- demo ---
np.random.seed(42)

print("=== Wasserstein Distance ===\n")

# 1D distributions
p1 = np.random.normal(0, 1, 1000)
p2 = np.random.normal(2, 1, 1000)
p3 = np.random.normal(0, 2, 1000)

print("--- 1D Wasserstein Distance ---")
print(f"  N(0,1) vs N(2,1): W1 = {wasserstein_1d(p1, p2):.3f} (shift=2)")
print(f"  N(0,1) vs N(0,2): W1 = {wasserstein_1d(p1, p3):.3f} (spread)")
print(f"  N(0,1) vs N(0,1): W1 = {wasserstein_1d(p1, p1):.3f} (same)")

# discrete optimal transport
print(f"\n--- Discrete Optimal Transport (Sinkhorn) ---")
n, m = 5, 5
# cost matrix (Euclidean between 1D points)
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([1, 2, 3, 4, 5], dtype=float)
C = np.abs(x[:, None] - y[None, :])
a = np.ones(n) / n  # uniform source
b = np.ones(m) / m  # uniform target

for reg in [1.0, 0.1, 0.01]:
    d = sinkhorn_distance(C, a, b, reg=reg)
    print(f"  reg={reg}: Sinkhorn distance = {d:.4f}")

print(f"\n  Exact W1 (shift by 1): 1.000")
print(f"  Wasserstein is a true metric on probability distributions.")
print(f"  Used in: WGAN, distribution comparison, domain adaptation.")
