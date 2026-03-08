"""K-Means Clustering — Lloyd's algorithm from scratch."""

import numpy as np

def k_means(X, k, max_iter=100):
    # random init
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for iteration in range(max_iter):
        # assign clusters
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        labels = dists.argmin(axis=0)

        # update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < 1e-6:
            print(f"  Converged at iteration {iteration}")
            break

    inertia = sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1).sum() for i in range(k))
    return labels, centroids, inertia

# --- demo ---
np.random.seed(42)
X = np.vstack([
    np.random.randn(30, 2) + [0, 0],
    np.random.randn(30, 2) + [5, 5],
    np.random.randn(30, 2) + [5, -3],
])

labels, centroids, inertia = k_means(X, k=3)
print(f"Centroids:\n{centroids.round(2)}")
print(f"Inertia: {inertia:.2f}")
print(f"Cluster sizes: {[int((labels==i).sum()) for i in range(3)]}")
