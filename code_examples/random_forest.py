"""Random Forest — ensemble of decision stumps with bagging."""

import numpy as np
from collections import Counter

class Stump:
    """Single-split decision tree (depth=1)."""
    def fit(self, X, y):
        best_gain, self.feat, self.thresh = -1, 0, 0
        parent_impurity = self._gini(y)
        for f in range(X.shape[1]):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left, right = y[X[:, f] <= t], y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_impurity - (len(left)/len(y)) * self._gini(left) \
                       - (len(right)/len(y)) * self._gini(right)
                if gain > best_gain:
                    best_gain, self.feat, self.thresh = gain, f, t
        left_y = y[X[:, self.feat] <= self.thresh]
        right_y = y[X[:, self.feat] > self.thresh]
        self.left_val = Counter(left_y).most_common(1)[0][0] if len(left_y) else 0
        self.right_val = Counter(right_y).most_common(1)[0][0] if len(right_y) else 0
        return self

    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thresh, self.left_val, self.right_val)

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

class RandomForest:
    def __init__(self, n_trees=50, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        n, d = X.shape
        mf = self.max_features or int(np.sqrt(d))
        for _ in range(self.n_trees):
            # bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            # random feature subset
            feats = np.random.choice(d, mf, replace=False)
            tree = Stump().fit(X[np.ix_(idx, feats)], y[idx])
            self.trees.append((tree, feats))
        return self

    def predict(self, X):
        votes = np.array([t.predict(X[:, feats]) for t, feats in self.trees])
        return np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

# --- demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(60, 4) + [1,0,1,0],
               np.random.randn(60, 4) + [-1,0,-1,0]])
y = np.array([0]*60 + [1]*60)

rf = RandomForest(n_trees=50).fit(X, y)
preds = rf.predict(X)
print(f"Random Forest accuracy: {(preds == y).mean():.1%}")
print(f"Trees: {rf.n_trees}, Features per split: {int(np.sqrt(4))}")
