"""Loss Functions — common ML losses and their gradients."""

import numpy as np

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def mse_grad(y, y_hat):
    return 2 * (y_hat - y) / len(y)

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def huber(y, y_hat, delta=1.0):
    diff = np.abs(y - y_hat)
    return np.mean(np.where(diff <= delta, 0.5 * diff**2, delta * diff - 0.5 * delta**2))

def cross_entropy(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def categorical_ce(y_onehot, p):
    p = np.clip(p, 1e-8, 1)
    return -np.mean(np.sum(y_onehot * np.log(p), axis=1))

def hinge_loss(y, scores):
    """y ∈ {-1, +1}"""
    return np.mean(np.maximum(0, 1 - y * scores))

def kl_divergence(p, q):
    p = np.clip(p, 1e-8, 1)
    q = np.clip(q, 1e-8, 1)
    return np.sum(p * np.log(p / q))

# --- demo ---
np.random.seed(42)
y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
y_hat = np.array([0.9, 0.2, 0.8, 0.6, 0.1])
y_reg = np.array([3.0, -1.0, 2.5, 0.5, -2.0])
y_pred = np.array([2.8, -0.5, 2.0, 1.0, -1.5])

print("=== Regression Losses ===")
print(f"  MSE:   {mse(y_reg, y_pred):.4f}")
print(f"  MAE:   {mae(y_reg, y_pred):.4f}")
print(f"  Huber: {huber(y_reg, y_pred):.4f}")

print("\n=== Classification Losses ===")
print(f"  Binary Cross-Entropy: {cross_entropy(y, y_hat):.4f}")
print(f"  Hinge Loss:           {hinge_loss(2*y-1, 2*y_hat-1):.4f}")

print("\n=== Information-Theoretic ===")
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])
print(f"  KL(p||uniform): {kl_divergence(p, q):.4f}")
