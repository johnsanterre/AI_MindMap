"""Dilated Convolution — exponentially growing receptive fields."""

import numpy as np

def conv1d(x, w, dilation=1):
    """1D convolution with dilation."""
    k = len(w)
    effective_k = (k - 1) * dilation + 1
    n = len(x)
    out_len = n - effective_k + 1
    out = np.zeros(out_len)
    for i in range(out_len):
        for j in range(k):
            out[i] += x[i + j * dilation] * w[j]
    return out

def dilated_stack(x, w_list, dilations):
    """Stack of dilated convolutions with residual connections."""
    h = x.copy()
    for w, d in zip(w_list, dilations):
        conv_out = conv1d(h[:len(h)], w, dilation=d)
        # pad to same length and add residual
        pad = len(h) - len(conv_out)
        conv_padded = np.concatenate([np.zeros(pad), conv_out])
        h = h + conv_padded
    return h

def receptive_field(kernel_size, dilations):
    """Calculate receptive field of dilated conv stack."""
    rf = 1
    for d in dilations:
        rf += (kernel_size - 1) * d
    return rf

# --- demo ---
np.random.seed(42)

print("=== Dilated Convolution ===\n")

# show dilation effect
x = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=float)
w = np.array([1, 1, 1], dtype=float)

print("Input:", x)
for d in [1, 2, 3]:
    out = conv1d(x, w, dilation=d)
    print(f"  dilation={d}: {out} (len={len(out)})")

# receptive field growth
print("\n--- Receptive Field Growth ---")
print(f"{'Layers':>7} {'Dilations':>20} {'Receptive Field':>16}")
for n_layers in [3, 5, 8, 10]:
    dilations = [2**i for i in range(n_layers)]
    rf = receptive_field(3, dilations)
    dil_str = str(dilations[:5]) + ("..." if n_layers > 5 else "")
    print(f"{n_layers:>7} {dil_str:>20} {rf:>16}")

# comparison: standard vs dilated
print(f"\n--- Standard vs Dilated (same param count) ---")
k = 3
for n_layers in [4, 6, 8]:
    rf_standard = 1 + n_layers * (k - 1)
    rf_dilated = receptive_field(k, [2**i for i in range(n_layers)])
    print(f"  {n_layers} layers: standard RF={rf_standard}, dilated RF={rf_dilated}")

print("\nUsed in WaveNet, TCN for audio/sequence modeling with long-range context.")
