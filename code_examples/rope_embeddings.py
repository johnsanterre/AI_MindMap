"""RoPE — Rotary Position Embeddings for transformer models."""

import numpy as np

def rope_frequencies(dim, max_len=128, base=10000):
    """Compute rotation frequencies for each dimension pair."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    positions = np.arange(max_len)
    angles = np.outer(positions, freqs)
    return angles

def apply_rope(x, angles):
    """Apply rotary embeddings to input tensor."""
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    cos = np.cos(angles[:x.shape[-2]])
    sin = np.sin(angles[:x.shape[-2]])
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def attention_with_rope(Q, K, V, angles):
    Q_rot = apply_rope(Q, angles)
    K_rot = apply_rope(K, angles)
    d_k = Q.shape[-1]
    scores = Q_rot @ K_rot.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights

# --- demo ---
np.random.seed(42)
seq_len = 8
d_model = 16

angles = rope_frequencies(d_model, max_len=128)

# show how RoPE encodes relative position
print("=== RoPE: Rotary Position Embeddings ===\n")
print("Rotation angles (first 4 freq pairs):")
print(f"{'Pos':>4}", end="")
for f in range(4):
    print(f"{'freq_'+str(f):>10}", end="")
print()
for pos in range(8):
    print(f"{pos:>4}", end="")
    for f in range(4):
        print(f"{angles[pos, f]:>10.4f}", end="")
    print()

# relative position property
print("\n--- Relative position dot product ---")
q = np.random.randn(seq_len, d_model)
k = q.copy()  # same vectors, different positions
q_rot = apply_rope(q, angles)
k_rot = apply_rope(k, angles)

print(f"{'Pos_q':>6} {'Pos_k':>6} {'Dot Product':>12}")
for i in [0, 2, 4]:
    for j in [0, 2, 4]:
        dot = q_rot[i] @ k_rot[j] / np.sqrt(d_model)
        print(f"{i:>6} {j:>6} {dot:>12.4f}")

print("\nDot products depend on relative position |i-j|, not absolute position.")
