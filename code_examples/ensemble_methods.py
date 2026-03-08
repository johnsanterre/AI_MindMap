"""Ensemble Methods — combining models for better predictions."""

import numpy as np

class SimpleTree:
    """Depth-1 decision stump."""
    def __init__(self):
        self.feature = 0
        self.threshold = 0
        self.left_val = 0
        self.right_val = 0

    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n) / n
        best_loss = np.inf
        for f in range(d):
            thresholds = np.percentile(X[:, f], np.arange(10, 100, 10))
            for t in thresholds:
                left = X[:, f] <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                l_val = np.average(y[left], weights=sample_weight[left])
                r_val = np.average(y[right], weights=sample_weight[right])
                pred = np.where(left, l_val, r_val)
                loss = np.average((y - pred)**2, weights=sample_weight)
                if loss < best_loss:
                    best_loss = loss
                    self.feature, self.threshold = f, t
                    self.left_val, self.right_val = l_val, r_val

    def predict(self, X):
        return np.where(X[:, self.feature] <= self.threshold, self.left_val, self.right_val)

def bagging(X, y, n_estimators=20):
    trees = []
    for _ in range(n_estimators):
        idx = np.random.choice(len(X), len(X), replace=True)
        tree = SimpleTree()
        tree.fit(X[idx], y[idx])
        trees.append(tree)
    return trees

def boosting(X, y, n_estimators=20, lr=0.1):
    trees = []
    pred = np.zeros(len(X))
    for _ in range(n_estimators):
        residual = y - pred
        tree = SimpleTree()
        tree.fit(X, residual)
        pred += lr * tree.predict(X)
        trees.append(tree)
    return trees, lr

def predict_ensemble(trees, X, method='bag', lr=0.1):
    if method == 'bag':
        return np.mean([t.predict(X) for t in trees], axis=0)
    else:
        return sum(lr * t.predict(X) for t in trees)

# --- demo ---
np.random.seed(42)
n = 200
X = np.random.randn(n, 4)
y = 3*X[:, 0] + X[:, 1]**2 - 2*X[:, 2] + np.random.randn(n) * 0.5

# train/test split
X_tr, X_te = X[:150], X[150:]
y_tr, y_te = y[:150], y[150:]

# single tree
single = SimpleTree()
single.fit(X_tr, y_tr)
mse_single = np.mean((y_te - single.predict(X_te))**2)

# bagging
bag_trees = bagging(X_tr, y_tr, 30)
mse_bag = np.mean((y_te - predict_ensemble(bag_trees, X_te, 'bag'))**2)

# boosting
boost_trees, lr = boosting(X_tr, y_tr, 50, lr=0.3)
mse_boost = np.mean((y_te - predict_ensemble(boost_trees, X_te, 'boost', lr))**2)

print("=== Ensemble Methods ===\n")
print(f"{'Method':>15} {'Test MSE':>10} {'Trees':>7}")
print("-" * 35)
print(f"{'Single Tree':>15} {mse_single:>10.4f} {1:>7}")
print(f"{'Bagging (30)':>15} {mse_bag:>10.4f} {30:>7}")
print(f"{'Boosting (50)':>15} {mse_boost:>10.4f} {50:>7}")
print(f"\nBagging reduces variance. Boosting reduces bias.")
