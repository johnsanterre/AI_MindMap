"""t-SNE — t-distributed Stochastic Neighbor Embedding (simplified)."""

import numpy as np

def pairwise_distances(X):
    sq = np.sum(X**2, axis=1)
    return sq[:, None] + sq[None, :] - 2 * X @ X.T

def p_joint(X, perplexity=30.0):
    D = pairwise_distances(X)
    n = X.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        # binary search for sigma
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2
            pi = np.exp(-D[i] / (2 * sigma**2))
            pi[i] = 0
            sum_pi = pi.sum()
            if sum_pi == 0:
                lo = sigma
                continue
            pi /= sum_pi
            entropy = -np.sum(pi[pi > 0] * np.log2(pi[pi > 0]))
            if entropy > np.log2(perplexity):
                hi = sigma
            else:
                lo = sigma
        P[i] = pi
    P = (P + P.T) / (2 * n)
    return np.maximum(P, 1e-12)

def tsne(X, n_components=2, perplexity=20.0, lr=100, n_iter=300):
    n = X.shape[0]
    P = p_joint(X, perplexity)
    Y = np.random.randn(n, n_components) * 0.01
    velocity = np.zeros_like(Y)

    for it in range(n_iter):
        D = pairwise_distances(Y)
        Q = 1 / (1 + D)
        np.fill_diagonal(Q, 0)
        Q /= Q.sum()
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            grad[i] = 4 * np.sum((PQ[i] * Q[i] * (1+D[i]))[:, None] * (Y[i] - Y), axis=0)

        velocity = 0.8 * velocity - lr * grad
        Y += velocity

        if it % 100 == 0:
            kl = np.sum(P * np.log(P / Q))
            print(f"  Iter {it:3d}  KL={kl:.4f}")

    return Y

# --- demo ---
np.random.seed(42)
# 3 clusters in 10D
X = np.vstack([
    np.random.randn(20, 10) + np.array([3]*5 + [0]*5),
    np.random.randn(20, 10) + np.array([0]*5 + [3]*5),
    np.random.randn(20, 10) + np.array([-3]*5 + [-3]*5),
])
labels = [0]*20 + [1]*20 + [2]*20

print("t-SNE: 10D → 2D")
Y = tsne(X, n_components=2, perplexity=15, n_iter=300, lr=50)

print(f"\nCluster centroids in 2D:")
for c in range(3):
    mask = np.array(labels) == c
    print(f"  Cluster {c}: ({Y[mask, 0].mean():.2f}, {Y[mask, 1].mean():.2f})")
