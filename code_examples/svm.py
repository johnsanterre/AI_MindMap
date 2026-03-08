"""Support Vector Machine — linear SVM with hinge loss and SGD."""

import numpy as np

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        for _ in range(self.epochs):
            for i in range(m):
                if y[i] * (X[i] @ self.w + self.b) >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]
        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b).astype(int)

    def margin(self):
        return 2.0 / np.linalg.norm(self.w)

# --- demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [2, 2],
               np.random.randn(50, 2) + [-2, -2]])
y = np.array([1]*50 + [-1]*50)

svm = LinearSVM(C=1.0).fit(X, y)
preds = svm.predict(X)
print(f"Accuracy: {(preds == y).mean():.1%}")
print(f"Weights: {svm.w.round(3)}, Bias: {svm.b:.3f}")
print(f"Margin width: {svm.margin():.3f}")
