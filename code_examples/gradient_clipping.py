"""Gradient Clipping — preventing exploding gradients."""

import numpy as np

def clip_by_value(grad, min_val=-1.0, max_val=1.0):
    return np.clip(grad, min_val, max_val)

def clip_by_norm(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad

def clip_by_global_norm(grads, max_norm=1.0):
    global_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
    if global_norm > max_norm:
        scale = max_norm / global_norm
        return [g * scale for g in grads], global_norm
    return grads, global_norm

# --- demo ---
np.random.seed(42)

# simulate exploding gradient scenario
print("=== Gradient Clipping Methods ===\n")
grad = np.random.randn(4) * 10  # large gradient
print(f"Original gradient:     {grad.round(3)}, norm={np.linalg.norm(grad):.3f}")
print(f"Clip by value [-1,1]:  {clip_by_value(grad).round(3)}, norm={np.linalg.norm(clip_by_value(grad)):.3f}")
print(f"Clip by norm (1.0):    {clip_by_norm(grad).round(3)}, norm={np.linalg.norm(clip_by_norm(grad)):.3f}")

# RNN-like scenario: gradient through time steps
print("\n--- Simulated RNN gradient explosion ---")
W = np.array([[1.5, 0.3], [-0.2, 1.4]])
grad = np.eye(2)
print(f"{'Step':>5} {'Grad norm (raw)':>16} {'Grad norm (clipped)':>20}")
for t in range(10):
    grad = grad @ W
    clipped = clip_by_norm(grad.flatten(), max_norm=5.0).reshape(2, 2)
    print(f"{t+1:>5} {np.linalg.norm(grad):>16.2f} {np.linalg.norm(clipped):>20.2f}")
