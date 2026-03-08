"""Backpropagation — step-by-step manual backprop through a computation graph."""

import numpy as np

# Computation graph: L = (1/n) * sum((y - sigmoid(X @ w + b))^2)
# Forward: z = Xw + b → a = sigmoid(z) → loss = MSE(a, y)
# Backward: dL/dw, dL/db via chain rule

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward(X, w, b, y):
    z = X @ w + b                       # linear
    a = sigmoid(z)                       # activation
    loss = np.mean((a - y) ** 2)         # MSE loss
    cache = (X, z, a, y)
    return loss, a, cache

def backward(cache):
    X, z, a, y = cache
    m = len(y)
    da = (2/m) * (a - y)                 # dL/da
    dz = da * a * (1 - a)               # da/dz = sigmoid'(z)
    dw = X.T @ dz                        # dz/dw = X
    db = dz.sum()                        # dz/db = 1
    return dw, db

# --- demo ---
np.random.seed(42)
X = np.random.randn(100, 3)
true_w = np.array([1.5, -2.0, 0.8])
y = sigmoid(X @ true_w + 0.5)
y = (y > 0.5).astype(float)

w = np.zeros(3)
b = 0.0
lr = 1.0

print("Step-by-step backprop training:\n")
for epoch in range(501):
    loss, preds, cache = forward(X, w, b, y)
    dw, db = backward(cache)
    w -= lr * dw
    b -= lr * db
    if epoch % 100 == 0:
        acc = ((preds > 0.5).astype(float) == y).mean()
        print(f"  Epoch {epoch:3d}  Loss={loss:.4f}  Acc={acc:.1%}")
        print(f"           dw={dw.round(4)}  db={db:.4f}")

print(f"\nFinal weights: {w.round(3)}")
