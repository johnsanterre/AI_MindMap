"""Logistic Regression — binary classification with gradient descent."""

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for epoch in range(epochs):
        p = sigmoid(X @ w + b)
        w -= lr * (1/m) * (X.T @ (p - y))
        b -= lr * (1/m) * (p - y).sum()
        if epoch % 200 == 0:
            loss = -(y * np.log(p + 1e-8) + (1-y) * np.log(1-p + 1e-8)).mean()
            print(f"  Epoch {epoch:4d}  Loss={loss:.4f}")
    return w, b

def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)

# --- demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [2, 2],
               np.random.randn(50, 2) + [-2, -2]])
y = np.array([1]*50 + [0]*50, dtype=float)

print("Training:")
w, b = logistic_regression(X, y, lr=0.1, epochs=1000)
preds = predict(X, w, b)
acc = (preds == y).mean()
print(f"\nWeights: {w.round(3)}, Bias: {b:.3f}")
print(f"Accuracy: {acc:.1%}")
