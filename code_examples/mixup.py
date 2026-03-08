"""Mixup — data augmentation by interpolating training examples."""

import numpy as np

def mixup_data(X, y_onehot, alpha=0.2):
    """Create mixed samples using Beta distribution."""
    lam = np.random.beta(alpha, alpha, size=len(X))
    lam = lam.reshape(-1, 1)
    indices = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[indices]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[indices]
    return X_mix, y_mix

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# --- demo ---
np.random.seed(42)
n_classes = 3
X = np.random.randn(12, 4)
y = np.array([0]*4 + [1]*4 + [2]*4)
y_onehot = np.eye(n_classes)[y]

X_mix, y_mix = mixup_data(X, y_onehot, alpha=0.4)

print("=== Mixup Data Augmentation ===\n")
print("Original samples (first 3):")
for i in range(3):
    print(f"  x={X[i].round(2)}, label={y_onehot[i]}")

print("\nMixed samples (first 3):")
for i in range(3):
    print(f"  x={X_mix[i].round(2)}, label={y_mix[i].round(3)}")

print("\nMixup creates 'between-class' examples that smooth decision boundaries.")
