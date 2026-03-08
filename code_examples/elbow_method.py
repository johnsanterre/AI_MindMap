"""Elbow Method — finding optimal number of clusters."""

import numpy as np

def kmeans(X, k, n_iter=50):
    centers = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(n_iter):
        dists = np.sum((X[:, None] - centers[None, :])**2, axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = X[labels == j].mean(axis=0)
    return labels, centers

def inertia(X, labels, centers):
    """Within-cluster sum of squares."""
    return sum(np.sum((X[labels == k] - centers[k])**2) for k in range(len(centers)))

def silhouette_score(X, labels):
    """Average silhouette coefficient."""
    n = len(X)
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        a = np.mean(np.sqrt(np.sum((X[same] - X[i])**2, axis=1))) if same.sum() > 1 else 0
        b = np.inf
        for k in np.unique(labels):
            if k == labels[i]:
                continue
            other = labels == k
            b = min(b, np.mean(np.sqrt(np.sum((X[other] - X[i])**2, axis=1))))
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return scores.mean()

# --- demo ---
np.random.seed(42)

# 4 true clusters
true_k = 4
data = np.vstack([np.random.randn(50, 2) + c for c in [[0,0], [5,0], [0,5], [5,5]]])

print("=== Elbow Method & Silhouette Score ===")
print(f"Data: {len(data)} points, true k={true_k}\n")

print(f"{'k':>3} {'Inertia':>10} {'Silhouette':>11} {'Δ Inertia':>11}")
print("-" * 38)
prev_inertia = None
for k in range(2, 9):
    labels, centers = kmeans(data, k)
    wcss = inertia(data, labels, centers)
    sil = silhouette_score(data, labels) if k < len(data) else 0
    delta = f"{prev_inertia - wcss:>11.1f}" if prev_inertia else "         -"
    print(f"{k:>3} {wcss:>10.1f} {sil:>11.3f} {delta}")
    prev_inertia = wcss

print(f"\nElbow: look for where inertia drop slows sharply (k={true_k}).")
print(f"Silhouette: higher is better, peak at k={true_k}.")
