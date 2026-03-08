"""Max Pooling — spatial downsampling for translation invariance."""

import numpy as np

def max_pool2d(x, pool_size=2, stride=2):
    H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = patch.max()
    return out

def avg_pool2d(x, pool_size=2, stride=2):
    H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = patch.mean()
    return out

def global_avg_pool(x):
    """(H, W, C) → (C,)."""
    return x.mean(axis=(0, 1))

# --- demo ---
np.random.seed(42)

x = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 7, 8],
    [4, 1, 3, 6]
], dtype=float)

print("=== Pooling Operations ===\n")
print("Input (4×4):")
print(x)

print(f"\nMax Pool 2×2, stride=2:")
print(max_pool2d(x))

print(f"\nAvg Pool 2×2, stride=2:")
print(avg_pool2d(x))

# translation invariance demo
print("\n--- Translation Invariance ---")
feature = np.zeros((8, 8))
feature[2:4, 2:4] = 1  # feature at position (2,2)
shifted = np.zeros((8, 8))
shifted[3:5, 3:5] = 1  # shifted by (1,1)

pool_orig = max_pool2d(feature, 2, 2)
pool_shift = max_pool2d(shifted, 2, 2)
print(f"Original feature at (2,2): max pool sum = {pool_orig.sum():.0f}")
print(f"Shifted feature at (3,3): max pool sum = {pool_shift.sum():.0f}")
print(f"Same output despite shift → translation invariance")

# global average pooling
x3d = np.random.randn(4, 4, 3)
gap = global_avg_pool(x3d)
print(f"\nGlobal Average Pool: (4,4,3) → ({len(gap)},) = {gap.round(3)}")
print("Replaces fully-connected layers in modern CNNs (fewer params).")
