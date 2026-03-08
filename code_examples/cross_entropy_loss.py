"""Cross-Entropy Loss — comparing loss functions for classification."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, targets):
    return -np.mean(np.sum(targets * np.log(probs + 1e-8), axis=1))

def binary_cross_entropy(p, y):
    return -np.mean(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8))

def focal_loss(probs, targets, gamma=2.0):
    pt = np.sum(probs * targets, axis=1)
    return -np.mean((1 - pt)**gamma * np.log(pt + 1e-8))

def hinge_loss(scores, y):
    """Multi-class hinge loss."""
    correct = scores[np.arange(len(y)), y]
    margins = np.maximum(0, scores - correct[:, None] + 1)
    margins[np.arange(len(y)), y] = 0
    return np.mean(np.sum(margins, axis=1))

# --- demo ---
np.random.seed(42)

n_classes = 4
n_samples = 100
logits = np.random.randn(n_samples, n_classes)
y = np.random.randint(0, n_classes, n_samples)
y_onehot = np.eye(n_classes)[y]
probs = softmax(logits)

print("=== Classification Loss Functions ===\n")
print(f"Cross-entropy:  {cross_entropy(probs, y_onehot):.4f}")
print(f"Focal (γ=2):    {focal_loss(probs, y_onehot, gamma=2.0):.4f}")
print(f"Focal (γ=0):    {focal_loss(probs, y_onehot, gamma=0.0):.4f}  (= cross-entropy)")
print(f"Hinge loss:     {hinge_loss(logits, y):.4f}")

# effect of confidence on loss
print("\n--- Cross-entropy vs confidence ---")
print(f"{'Prob of true':>14} {'CE Loss':>10} {'Focal(γ=2)':>12}")
for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    ce = -np.log(p)
    fl = -(1-p)**2 * np.log(p)
    print(f"{p:>14.2f} {ce:>10.4f} {fl:>12.4f}")

print("\nFocal loss down-weights easy examples, focusing on hard ones.")
