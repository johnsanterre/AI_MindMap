"""Naive Bayes — Gaussian Naive Bayes classifier."""

import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = {}
        for c in self.classes:
            Xc = X[y == c]
            self.params[c] = {
                'mean': Xc.mean(axis=0),
                'var': Xc.var(axis=0) + 1e-8,
                'prior': len(Xc) / len(y),
            }
        return self

    def _log_prob(self, x, mean, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (x - mean)**2 / var)

    def predict(self, X):
        preds = []
        for x in X:
            scores = {}
            for c in self.classes:
                p = self.params[c]
                scores[c] = np.log(p['prior']) + self._log_prob(x, p['mean'], p['var'])
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

# --- demo ---
np.random.seed(42)
X = np.vstack([
    np.random.randn(40, 2) + [2, 0],
    np.random.randn(40, 2) + [-2, 0],
    np.random.randn(40, 2) + [0, 3],
])
y = np.array([0]*40 + [1]*40 + [2]*40)

# shuffle and split
idx = np.random.permutation(120)
X, y = X[idx], y[idx]
X_train, X_test = X[:90], X[90:]
y_train, y_test = y[:90], y[90:]

nb = GaussianNB().fit(X_train, y_train)
preds = nb.predict(X_test)
acc = (preds == y_test).mean()
print(f"Accuracy: {acc:.1%}")
print(f"Predictions: {preds}")
print(f"Actual:      {y_test}")
