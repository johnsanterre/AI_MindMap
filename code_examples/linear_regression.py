"""Linear Regression — gradient descent from scratch."""

import numpy as np

def linear_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for epoch in range(epochs):
        pred = X @ w + b
        error = pred - y
        w -= lr * (2/m) * (X.T @ error)
        b -= lr * (2/m) * error.sum()
        if epoch % 200 == 0:
            mse = (error ** 2).mean()
            print(f"  Epoch {epoch:4d}  MSE={mse:.4f}")
    return w, b

# --- demo ---
np.random.seed(42)
X = np.random.randn(100, 3)
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 3.0 + np.random.randn(100) * 0.1

print("Training:")
w, b = linear_regression(X, y, lr=0.05, epochs=1000)
print(f"\nLearned weights: {w.round(3)}")
print(f"Learned bias:    {b:.3f}")
print(f"True weights:    {true_w}")
print(f"True bias:       3.0")
