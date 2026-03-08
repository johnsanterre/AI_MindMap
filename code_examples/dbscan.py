"""DBSCAN — density-based clustering that finds arbitrary-shape clusters."""

import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = np.full(n, -1)  # -1 = unvisited/noise
    cluster_id = 0
    dists = np.sqrt(np.sum((X[:, None] - X[None, :])**2, axis=2))

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = np.where(dists[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -2  # noise
            continue
        # expand cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if labels[q] == -2:
                labels[q] = cluster_id  # noise becomes border
            elif labels[q] == -1:
                labels[q] = cluster_id
                q_neighbors = np.where(dists[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors.tolist())
            j += 1
        cluster_id += 1

    labels[labels == -2] = -1  # mark noise as -1
    return labels

# --- demo ---
np.random.seed(42)

# three blobs + noise
blob1 = np.random.randn(30, 2) * 0.4 + [0, 0]
blob2 = np.random.randn(30, 2) * 0.4 + [3, 3]
blob3 = np.random.randn(30, 2) * 0.4 + [6, 0]
noise = np.random.uniform(-2, 8, size=(10, 2))
X = np.vstack([blob1, blob2, blob3, noise])

labels = dbscan(X, eps=1.0, min_samples=5)
n_clusters = len(set(labels) - {-1})
n_noise = (labels == -1).sum()

print(f"=== DBSCAN Clustering ===")
print(f"Points: {len(X)}")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
for c in range(n_clusters):
    print(f"  Cluster {c}: {(labels == c).sum()} points")
