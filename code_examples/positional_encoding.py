"""Positional Encoding — sinusoidal encodings used in Transformers."""

import numpy as np

def sinusoidal_encoding(seq_len, d_model):
    """Create positional encoding matrix as in 'Attention Is All You Need'."""
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE

def relative_distance(PE, pos_i, pos_j):
    """Show that dot product encodes relative position."""
    return PE[pos_i] @ PE[pos_j]

# --- demo ---
seq_len, d_model = 20, 16
PE = sinusoidal_encoding(seq_len, d_model)

print(f"Positional Encoding: {seq_len} positions × {d_model} dims\n")

print("First 4 positions (first 8 dims):")
for i in range(4):
    print(f"  pos {i}: {PE[i, :8].round(3)}")

print(f"\nDot product (measures similarity):")
print(f"  pos 0 · pos 0 = {relative_distance(PE, 0, 0):.3f}")
print(f"  pos 0 · pos 1 = {relative_distance(PE, 0, 1):.3f}")
print(f"  pos 0 · pos 5 = {relative_distance(PE, 0, 5):.3f}")
print(f"  pos 0 · pos 10 = {relative_distance(PE, 0, 10):.3f}")
print(f"  pos 5 · pos 6 = {relative_distance(PE, 5, 6):.3f}  (same distance as 0·1)")

print(f"\nProperty: nearby positions have similar encodings")
dists = [relative_distance(PE, 0, i) for i in range(seq_len)]
print(f"  Similarity to pos 0: {[f'{d:.2f}' for d in dists[:10]]}")
