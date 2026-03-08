"""Label Smoothing — regularization by softening hard targets."""

import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def cross_entropy(logits, targets):
    probs = softmax(logits)
    return -np.mean(np.sum(targets * np.log(probs + 1e-8), axis=1))

def smooth_labels(y_onehot, n_classes, alpha=0.1):
    """Replace hard labels with smoothed version."""
    return y_onehot * (1 - alpha) + alpha / n_classes

def train_step(W, b, X, y_hard, y_smooth, lr=0.01):
    logits = X @ W + b
    probs = softmax(logits)
    # hard label loss
    hard_loss = -np.mean(np.sum(y_hard * np.log(probs + 1e-8), axis=1))
    # smooth label loss
    smooth_loss = -np.mean(np.sum(y_smooth * np.log(probs + 1e-8), axis=1))
    # gradient using smooth labels
    grad = (probs - y_smooth) / len(X)
    W -= lr * X.T @ grad
    b -= lr * grad.sum(axis=0)
    return hard_loss, smooth_loss

# --- demo ---
np.random.seed(42)
n_classes = 5
X = np.random.randn(100, 8)
y = np.random.randint(0, n_classes, 100)
y_hard = np.eye(n_classes)[y]
y_smooth = smooth_labels(y_hard, n_classes, alpha=0.1)

print("Hard labels (sample):", y_hard[0])
print("Smooth labels (α=0.1):", y_smooth[0].round(3))

W = np.random.randn(8, n_classes) * 0.1
b = np.zeros(n_classes)

print(f"\n{'Epoch':>6} {'Hard Loss':>10} {'Smooth Loss':>12}")
print("-" * 32)
for epoch in range(201):
    hl, sl = train_step(W, b, X, y_hard, y_smooth, lr=0.1)
    if epoch % 50 == 0:
        print(f"{epoch:>6} {hl:>10.4f} {sl:>12.4f}")

# check confidence
logits = X @ W + b
probs = softmax(logits)
print(f"\nAvg max confidence: {probs.max(axis=1).mean():.3f}")
print("(Label smoothing prevents overconfident predictions)")
