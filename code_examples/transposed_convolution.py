"""Transposed Convolution — learnable upsampling for decoder networks."""

import numpy as np

def conv1d(x, w, stride=1):
    """Forward 1D convolution."""
    n, k = len(x), len(w)
    out_len = (n - k) // stride + 1
    return np.array([np.sum(x[i*stride:i*stride+k] * w) for i in range(out_len)])

def transposed_conv1d(x, w, stride=1):
    """Transposed (fractionally-strided) 1D convolution."""
    k = len(w)
    out_len = (len(x) - 1) * stride + k
    out = np.zeros(out_len)
    for i, val in enumerate(x):
        out[i*stride:i*stride+k] += val * w
    return out

def conv2d_transpose(x, W, stride=1):
    """Transposed 2D convolution: (H,W) * (kH,kW) → (H',W')."""
    H, W_in = x.shape
    kH, kW = W.shape
    H_out = (H - 1) * stride + kH
    W_out = (W_in - 1) * stride + kW
    out = np.zeros((H_out, W_out))
    for i in range(H):
        for j in range(W_in):
            out[i*stride:i*stride+kH, j*stride:j*stride+kW] += x[i, j] * W
    return out

# --- demo ---
np.random.seed(42)

print("=== Transposed Convolution ===\n")

# 1D example
x = np.array([1.0, 2.0, 3.0])
w = np.array([1.0, 0.5])

fwd = conv1d(x, w, stride=1)
trans = transposed_conv1d(fwd, w, stride=1)

print("1D Example:")
print(f"  Input:         {x}")
print(f"  Forward conv:  {fwd}")
print(f"  Transposed:    {trans}")

# stride=2 upsampling
print(f"\n  Stride=2 upsampling:")
x2 = np.array([1.0, 2.0, 3.0])
up = transposed_conv1d(x2, np.array([1.0, 1.0, 0.5]), stride=2)
print(f"  Input (len={len(x2)}):  {x2}")
print(f"  Output (len={len(up)}): {up}")

# 2D example
print(f"\n2D Example:")
x2d = np.array([[1.0, 2.0], [3.0, 4.0]])
w2d = np.array([[1.0, 0.5], [0.5, 0.25]])
out2d = conv2d_transpose(x2d, w2d, stride=2)
print(f"  Input: {x2d.shape} → Output: {out2d.shape}")
print(f"  Used in: U-Net decoder, GAN generator, semantic segmentation")

# checkerboard artifacts
print(f"\n  Stride=2 can cause checkerboard artifacts in generated images.")
print(f"  Fix: use nearest-neighbor upsample + regular conv instead.")
