"""KL Divergence — measuring distance between probability distributions."""

import numpy as np

def kl_divergence(p, q):
    """KL(P || Q) = sum p(x) log(p(x)/q(x))."""
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-10)))

def js_divergence(p, q):
    """Jensen-Shannon divergence: symmetric version of KL."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def gaussian_kl(mu1, sig1, mu2, sig2):
    """KL between two univariate Gaussians."""
    return np.log(sig2/sig1) + (sig1**2 + (mu1-mu2)**2)/(2*sig2**2) - 0.5

# --- demo ---
print("=== KL Divergence ===\n")

# discrete distributions
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
r = np.array([0.1, 0.2, 0.3, 0.4])       # reversed

print("Discrete distributions:")
print(f"  P = {p}")
print(f"  Q = {q} (uniform)")
print(f"  R = {r} (reversed P)\n")

print(f"  KL(P || Q) = {kl_divergence(p, q):.4f}")
print(f"  KL(Q || P) = {kl_divergence(q, p):.4f}  (asymmetric!)")
print(f"  KL(P || R) = {kl_divergence(p, r):.4f}")
print(f"  JS(P, Q)   = {js_divergence(p, q):.4f}  (symmetric)")
print(f"  JS(P, R)   = {js_divergence(p, r):.4f}")

# Gaussian KL
print("\n--- Gaussian KL divergence ---")
print(f"{'μ₁':>5} {'σ₁':>5} {'μ₂':>5} {'σ₂':>5} {'KL(1||2)':>10}")
for mu1, s1, mu2, s2 in [(0,1,0,1), (0,1,1,1), (0,1,0,2), (2,0.5,0,1)]:
    print(f"{mu1:>5} {s1:>5} {mu2:>5} {s2:>5} {gaussian_kl(mu1,s1,mu2,s2):>10.4f}")

print("\nKL=0 when distributions are identical.")
print("KL divergence is the extra bits needed to code P using Q's code.")
