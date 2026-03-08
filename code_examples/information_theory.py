"""Information Theory — entropy, mutual information, KL divergence."""

import numpy as np

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def joint_entropy(pxy):
    return entropy(pxy.ravel())

def conditional_entropy(pxy):
    """H(Y|X) = H(X,Y) - H(X)"""
    px = pxy.sum(axis=1)
    return joint_entropy(pxy) - entropy(px)

def mutual_information(pxy):
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    return entropy(px) + entropy(py) - joint_entropy(pxy)

def kl_divergence(p, q):
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def cross_entropy_dist(p, q):
    mask = p > 0
    return -np.sum(p[mask] * np.log2(q[mask]))

# --- demo ---
# 1) Entropy of different distributions
print("=== Entropy ===")
for name, p in [("Fair coin", [0.5, 0.5]),
                ("Biased coin (90/10)", [0.9, 0.1]),
                ("Fair die", [1/6]*6),
                ("Peaked", [0.01, 0.01, 0.96, 0.01, 0.01])]:
    p = np.array(p)
    print(f"  {name:>25}: H = {entropy(p):.4f} bits")

# 2) Mutual information
print("\n=== Mutual Information ===")
# joint distribution: X = weather, Y = umbrella
pxy = np.array([[0.35, 0.05],   # sunny: no umbrella, umbrella
                [0.10, 0.20],   # cloudy
                [0.05, 0.25]])  # rainy
print(f"  H(Weather) = {entropy(pxy.sum(axis=1)):.4f} bits")
print(f"  H(Umbrella) = {entropy(pxy.sum(axis=0)):.4f} bits")
print(f"  I(Weather; Umbrella) = {mutual_information(pxy):.4f} bits")
print(f"  H(Umbrella|Weather) = {conditional_entropy(pxy):.4f} bits")

# 3) KL divergence
print("\n=== KL Divergence ===")
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])
print(f"  KL(p||uniform) = {kl_divergence(p, q):.4f} bits")
print(f"  KL(uniform||p) = {kl_divergence(q, p):.4f} bits (asymmetric!)")
