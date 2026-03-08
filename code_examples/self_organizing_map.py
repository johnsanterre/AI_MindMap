"""Self-Organizing Map — unsupervised topology-preserving dimensionality reduction."""

import numpy as np

class SOM:
    def __init__(self, grid_size, input_dim):
        self.h, self.w = grid_size
        self.weights = np.random.randn(self.h, self.w, input_dim) * 0.1

    def find_bmu(self, x):
        """Find best matching unit."""
        dists = np.sum((self.weights - x)**2, axis=2)
        idx = np.unravel_index(dists.argmin(), dists.shape)
        return idx

    def neighborhood(self, bmu, sigma):
        """Gaussian neighborhood function."""
        y, x = np.meshgrid(range(self.h), range(self.w), indexing='ij')
        d2 = (y - bmu[0])**2 + (x - bmu[1])**2
        return np.exp(-d2 / (2 * sigma**2))

    def train(self, X, n_epochs=100, lr0=0.5, sigma0=None):
        if sigma0 is None:
            sigma0 = max(self.h, self.w) / 2
        for epoch in range(n_epochs):
            t = epoch / n_epochs
            lr = lr0 * (1 - t)
            sigma = sigma0 * (1 - t) + 0.5
            for x in X[np.random.permutation(len(X))]:
                bmu = self.find_bmu(x)
                h = self.neighborhood(bmu, sigma)[:, :, None]
                self.weights += lr * h * (x - self.weights)

    def map_data(self, X):
        return np.array([self.find_bmu(x) for x in X])

# --- demo ---
np.random.seed(42)

# 3 clusters in 4D
c1 = np.random.randn(30, 4) + [3, 0, 0, 0]
c2 = np.random.randn(30, 4) + [0, 3, 0, 0]
c3 = np.random.randn(30, 4) + [0, 0, 3, 0]
X = np.vstack([c1, c2, c3])
labels = np.array([0]*30 + [1]*30 + [2]*30)

som = SOM((5, 5), 4)
som.train(X, n_epochs=50)
mapped = som.map_data(X)

print("=== Self-Organizing Map ===")
print(f"Input: {X.shape[1]}D, Grid: 5×5, Samples: {len(X)}\n")

print("Cluster map positions (mode per cluster):")
for c in range(3):
    positions = mapped[labels == c]
    unique, counts = np.unique(positions, axis=0, return_counts=True)
    mode = unique[counts.argmax()]
    print(f"  Cluster {c}: most common position = ({mode[0]}, {mode[1]})")

# check topology preservation
print(f"\nTopology check (nearby clusters should map nearby):")
centers = np.array([mapped[labels == c].mean(axis=0) for c in range(3)])
for i in range(3):
    for j in range(i+1, 3):
        d = np.sqrt(np.sum((centers[i] - centers[j])**2))
        print(f"  Cluster {i} ↔ {j}: grid distance = {d:.2f}")
