"""Mutual Information — measuring shared information between variables."""

import numpy as np

def entropy_discrete(x, bins=20):
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def joint_entropy(x, y, bins=20):
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def mutual_information(x, y, bins=20):
    """MI(X;Y) = H(X) + H(Y) - H(X,Y)."""
    return entropy_discrete(x, bins) + entropy_discrete(y, bins) - joint_entropy(x, y, bins)

def normalized_mi(x, y, bins=20):
    """NMI = MI / sqrt(H(X) * H(Y))."""
    mi = mutual_information(x, y, bins)
    hx = entropy_discrete(x, bins)
    hy = entropy_discrete(y, bins)
    return mi / np.sqrt(hx * hy + 1e-10)

# --- demo ---
np.random.seed(42)
n = 5000

print("=== Mutual Information ===\n")

# independent variables
x_ind = np.random.randn(n)
y_ind = np.random.randn(n)
mi_ind = mutual_information(x_ind, y_ind)
print(f"Independent:     MI = {mi_ind:.4f} (should be ≈ 0)")

# linearly dependent
x_lin = np.random.randn(n)
y_lin = 0.8 * x_lin + np.random.randn(n) * 0.3
mi_lin = mutual_information(x_lin, y_lin)
print(f"Linear (r=0.8):  MI = {mi_lin:.4f}")

# nonlinear dependence (Pearson misses this)
x_nl = np.random.randn(n)
y_nl = x_nl**2 + np.random.randn(n) * 0.1
pearson = np.abs(np.corrcoef(x_nl, y_nl)[0,1])
mi_nl = mutual_information(x_nl, y_nl)
print(f"Quadratic:       MI = {mi_nl:.4f}, |Pearson| = {pearson:.4f}")

# deterministic
x_det = np.random.randn(n)
y_det = np.sin(x_det)
mi_det = mutual_information(x_det, y_det)
print(f"Deterministic:   MI = {mi_det:.4f}")

# feature selection
print(f"\n--- Feature Selection via MI ---")
X = np.random.randn(n, 5)
y = 2*X[:, 0] + X[:, 1]**2 + np.random.randn(n) * 0.1
print(f"y = 2*x0 + x1² + noise")
for i in range(5):
    mi = mutual_information(X[:, i], y)
    nmi = normalized_mi(X[:, i], y)
    print(f"  Feature x{i}: MI={mi:.3f}, NMI={nmi:.3f}")
