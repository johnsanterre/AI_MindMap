"""Isolation Forest — anomaly detection via random partitioning."""

import numpy as np

class IsolationTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n, d = X.shape
        self.size = n
        if n <= 1 or depth >= self.max_depth:
            self.is_leaf = True
            return self
        self.is_leaf = False
        self.feature = np.random.randint(d)
        lo, hi = X[:, self.feature].min(), X[:, self.feature].max()
        if lo == hi:
            self.is_leaf = True
            return self
        self.threshold = np.random.uniform(lo, hi)
        left_mask = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            return depth + _c(self.size)
        if x[self.feature] < self.threshold:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)

def _c(n):
    """Average path length of unsuccessful BST search."""
    if n <= 1: return 0
    return 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n

class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256, max_depth=10):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth

    def fit(self, X):
        self.trees = []
        n = len(X)
        for _ in range(self.n_trees):
            idx = np.random.choice(n, min(self.sample_size, n), replace=False)
            tree = IsolationTree(self.max_depth).fit(X[idx])
            self.trees.append(tree)
        return self

    def score(self, X):
        avg_lengths = np.array([
            np.mean([t.path_length(x) for t in self.trees]) for x in X
        ])
        scores = 2 ** (-avg_lengths / _c(self.sample_size))
        return scores

# --- demo ---
np.random.seed(42)
normal = np.random.randn(200, 2)
anomalies = np.random.uniform(-4, 4, size=(10, 2))
X = np.vstack([normal, anomalies])
labels = np.array([0]*200 + [1]*10)

iforest = IsolationForest(n_trees=100, sample_size=128).fit(X)
scores = iforest.score(X)

threshold = np.percentile(scores, 95)
predictions = (scores > threshold).astype(int)

print("=== Isolation Forest Anomaly Detection ===")
print(f"Normal points: 200, Anomalies: 10")
print(f"Avg score (normal):  {scores[:200].mean():.3f}")
print(f"Avg score (anomaly): {scores[200:].mean():.3f}")
print(f"Detected anomalies: {predictions.sum()} (threshold={threshold:.3f})")
tp = (predictions[200:] == 1).sum()
print(f"True positives: {tp}/10")
