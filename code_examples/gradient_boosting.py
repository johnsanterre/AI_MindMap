"""Gradient Boosting — regression with decision stumps."""

import numpy as np

class Stump:
    def fit(self, X, residuals):
        best_mse = float('inf')
        self.feat = 0
        self.thresh = 0
        self.left_val = 0
        self.right_val = 0
        for f in range(X.shape[1]):
            for t in np.percentile(X[:, f], np.arange(10, 100, 10)):
                left = residuals[X[:, f] <= t]
                right = residuals[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (left.var() * len(left) + right.var() * len(right)) / len(residuals)
                if mse < best_mse:
                    best_mse = mse
                    self.feat, self.thresh = f, t
                    self.left_val, self.right_val = left.mean(), right.mean()
        return self

    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thresh, self.left_val, self.right_val)

def gradient_boost(X, y, n_trees=50, lr=0.1):
    base = y.mean()
    trees = []
    pred = np.full(len(y), base)
    for i in range(n_trees):
        residuals = y - pred
        tree = Stump().fit(X, residuals)
        trees.append(tree)
        pred += lr * tree.predict(X)
        if i % 10 == 0:
            mse = np.mean((y - pred) ** 2)
            print(f"  Tree {i:3d}  MSE={mse:.4f}")
    return base, trees, lr

def gb_predict(X, base, trees, lr):
    pred = np.full(len(X), base)
    for tree in trees:
        pred += lr * tree.predict(X)
    return pred

# --- demo ---
np.random.seed(42)
X = np.random.randn(200, 3)
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.random.randn(200) * 0.1

base, trees, lr = gradient_boost(X, y, n_trees=50, lr=0.3)
preds = gb_predict(X, base, trees, lr)
print(f"\nFinal MSE: {np.mean((y - preds)**2):.4f}")
print(f"R²: {1 - np.sum((y-preds)**2)/np.sum((y-y.mean())**2):.3f}")
