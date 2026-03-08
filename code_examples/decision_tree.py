"""Decision Tree — classification tree with information gain."""

import numpy as np
from collections import Counter

def entropy(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

def info_gain(X, y, feature, threshold):
    left = y[X[:, feature] <= threshold]
    right = y[X[:, feature] > threshold]
    if len(left) == 0 or len(right) == 0:
        return 0
    p = len(left) / len(y)
    return entropy(y) - p * entropy(left) - (1-p) * entropy(right)

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            self.value = Counter(y).most_common(1)[0][0]
            self.feature = None
            return self
        best_gain, best_feat, best_thresh = -1, None, None
        for f in range(X.shape[1]):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                g = info_gain(X, y, f, t)
                if g > best_gain:
                    best_gain, best_feat, best_thresh = g, f, t
        if best_gain <= 0:
            self.value = Counter(y).most_common(1)[0][0]
            self.feature = None
            return self
        self.feature = best_feat
        self.threshold = best_thresh
        left_mask = X[:, best_feat] <= best_thresh
        self.left = DecisionTree(self.max_depth).fit(X[left_mask], y[left_mask], depth+1)
        self.right = DecisionTree(self.max_depth).fit(X[~left_mask], y[~left_mask], depth+1)
        return self

    def predict_one(self, x):
        if self.feature is None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# --- demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [1, 1],
               np.random.randn(50, 2) + [-1, -1]])
y = np.array([0]*50 + [1]*50)

tree = DecisionTree(max_depth=3).fit(X, y)
preds = tree.predict(X)
print(f"Accuracy: {(preds == y).mean():.1%}")
print(f"Root split: feature {tree.feature}, threshold {tree.threshold:.3f}")
