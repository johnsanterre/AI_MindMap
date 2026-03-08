"""Mean Shift — mode-seeking clustering without specifying k."""

import numpy as np

def gaussian_kernel(dist, bandwidth):
    return np.exp(-0.5 * (dist / bandwidth)**2)

def mean_shift(X, bandwidth=1.0, n_iter=50, tol=1e-4):
    points = X.copy()
    for _ in range(n_iter):
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            dists = np.linalg.norm(X - points[i], axis=1)
            weights = gaussian_kernel(dists, bandwidth)
            new_points[i] = np.average(X, weights=weights, axis=0)
        shift = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        if shift < tol:
            break
    # merge nearby modes
    labels = np.full(len(points), -1)
    cluster_id = 0
    for i in range(len(points)):
        if labels[i] >= 0:
            continue
        labels[i] = cluster_id
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < bandwidth / 2:
                labels[j] = cluster_id
        cluster_id += 1
    return labels, points

# --- demo ---
np.random.seed(42)

# 3 clusters
c1 = np.random.randn(40, 2) * 0.5 + [0, 0]
c2 = np.random.randn(40, 2) * 0.5 + [4, 0]
c3 = np.random.randn(40, 2) * 0.5 + [2, 3]
X = np.vstack([c1, c2, c3])

labels, modes = mean_shift(X, bandwidth=1.5)
n_clusters = len(np.unique(labels))

print("=== Mean Shift Clustering ===")
print(f"Points: {len(X)}, Bandwidth: 1.5")
print(f"Clusters found: {n_clusters}\n")

for c in range(n_clusters):
    mask = labels == c
    center = X[mask].mean(axis=0)
    print(f"  Cluster {c}: {mask.sum()} points, center=({center[0]:.2f}, {center[1]:.2f})")

print("\nMean shift automatically determines the number of clusters.")
