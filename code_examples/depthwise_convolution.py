"""Depthwise Separable Convolution — efficient convolution factorization."""

import numpy as np

def standard_conv2d(x, W, stride=1):
    """Standard convolution: (H,W,C_in) * (kH,kW,C_in,C_out) → (H',W',C_out)."""
    H, W_in, C_in = x.shape
    kH, kW, _, C_out = W.shape
    H_out = (H - kH) // stride + 1
    W_out = (W_in - kW) // stride + 1
    out = np.zeros((H_out, W_out, C_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
            out[i, j] = np.sum(patch[:, :, :, None] * W, axis=(0, 1, 2))
    return out

def depthwise_conv2d(x, W_dw, stride=1):
    """Depthwise: each channel convolved separately. W_dw: (kH,kW,C)."""
    H, W_in, C = x.shape
    kH, kW, _ = W_dw.shape
    H_out = (H - kH) // stride + 1
    W_out = (W_in - kW) // stride + 1
    out = np.zeros((H_out, W_out, C))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
            out[i, j] = np.sum(patch * W_dw, axis=(0, 1))
    return out

def pointwise_conv(x, W_pw):
    """1x1 convolution: (H,W,C_in) * (C_in,C_out) → (H,W,C_out)."""
    return x @ W_pw

def depthwise_separable_conv(x, W_dw, W_pw, stride=1):
    h = depthwise_conv2d(x, W_dw, stride)
    return pointwise_conv(h, W_pw)

# --- demo ---
np.random.seed(42)
H, W, C_in, C_out = 8, 8, 16, 32
k = 3

x = np.random.randn(H, W, C_in)
W_std = np.random.randn(k, k, C_in, C_out) * 0.1
W_dw = np.random.randn(k, k, C_in) * 0.1
W_pw = np.random.randn(C_in, C_out) * 0.1

out_std = standard_conv2d(x, W_std)
out_sep = depthwise_separable_conv(x, W_dw, W_pw)

params_std = k * k * C_in * C_out
params_sep = k * k * C_in + C_in * C_out

print("=== Depthwise Separable Convolution ===")
print(f"Input: ({H},{W},{C_in}), Kernel: {k}×{k}, Output channels: {C_out}\n")
print(f"Standard conv output:  {out_std.shape}, params: {params_std}")
print(f"Separable conv output: {out_sep.shape}, params: {params_sep}")
print(f"Parameter reduction: {params_sep/params_std:.1%} ({params_std/params_sep:.1f}× fewer)")
print(f"\nFLOPs ratio ≈ 1/C_out + 1/k² = {1/C_out + 1/k**2:.3f}")
print("Used in MobileNet, EfficientNet for mobile/edge deployment.")
