"""Layer Normalization — normalizing across features, not batch."""

import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = gamma * x_norm + beta
    return x_norm

def batch_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = gamma * x_norm + beta
    return x_norm

def rms_norm(x, gamma=None, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    if gamma is not None:
        x_norm = gamma * x_norm
    return x_norm

# --- demo ---
np.random.seed(42)

# simulate transformer hidden states: (batch=4, seq=3, dim=8)
X = np.random.randn(4, 3, 8) * np.array([1, 5, 0.1]).reshape(1, 3, 1)

print("=== Normalization Comparison ===\n")
print("Input stats per sequence position:")
for t in range(3):
    print(f"  pos {t}: mean={X[:, t].mean():.3f}, std={X[:, t].std():.3f}")

# Layer norm (per-token)
ln = layer_norm(X)
print("\nAfter Layer Norm (per token):")
for t in range(3):
    print(f"  pos {t}: mean={ln[:, t].mean():.4f}, std={ln[:, t].std():.4f}")

# RMS norm
rn = rms_norm(X)
print("\nAfter RMS Norm (per token):")
for t in range(3):
    print(f"  pos {t}: rms={np.sqrt(np.mean(rn[:, t]**2)):.4f}")

# batch norm vs layer norm with varying batch sizes
print("\n--- Batch size sensitivity ---")
print(f"{'Batch Size':>11} {'BN Var':>10} {'LN Var':>10}")
for bs in [2, 4, 8, 32]:
    x = np.random.randn(bs, 8) * 3 + 2
    bn = batch_norm(x)
    ln2 = layer_norm(x)
    print(f"{bs:>11} {bn.var():.6f} {ln2.var():.6f}")

print("\nLayer norm is batch-size independent — preferred for transformers.")
