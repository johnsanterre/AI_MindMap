"""Perceptron — the original learning algorithm."""

import numpy as np

class Perceptron:
    def __init__(self, lr=0.1):
        self.lr = lr

    def fit(self, X, y, epochs=100):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors = []
        for epoch in range(epochs):
            errs = 0
            for xi, yi in zip(X, y):
                pred = 1 if (xi @ self.w + self.b) >= 0 else -1
                if pred != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errs += 1
            self.errors.append(errs)
            if errs == 0:
                print(f"  Converged at epoch {epoch}")
                break
        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b).astype(int)

# --- demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [2, 2],
               np.random.randn(50, 2) + [-2, -2]])
y = np.array([1]*50 + [-1]*50)

p = Perceptron(lr=0.1).fit(X, y)
preds = p.predict(X)
acc = (preds == y).mean()
print(f"Weights: {p.w.round(3)}, Bias: {p.b:.3f}")
print(f"Accuracy: {acc:.1%}")
print(f"Error history (first 10): {p.errors[:10]}")
