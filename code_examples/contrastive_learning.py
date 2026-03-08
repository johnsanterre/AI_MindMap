"""Contrastive Learning — SimCLR-style NT-Xent loss."""

import numpy as np

def augment(x, noise=0.3):
    """Simple augmentation: add noise."""
    return x + np.random.randn(*x.shape) * noise

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    batch_size = len(z_i)
    z = np.vstack([z_i, z_j])  # 2N x d
    z = normalize(z)
    sim = z @ z.T / temperature
    # mask out self-similarity
    np.fill_diagonal(sim, -1e9)

    labels = np.concatenate([np.arange(batch_size, 2*batch_size),
                             np.arange(batch_size)])
    # cross-entropy for each sample
    loss = 0
    for i in range(2 * batch_size):
        exp_sim = np.exp(sim[i] - sim[i].max())
        loss -= np.log(exp_sim[labels[i]] / exp_sim.sum() + 1e-8)
    return loss / (2 * batch_size)

def train_encoder(X, d_proj=8, epochs=200, lr=0.01, temp=0.5):
    """Train a simple linear encoder with contrastive loss."""
    d = X.shape[1]
    W = np.random.randn(d, d_proj) * 0.1

    for epoch in range(epochs):
        # create positive pairs via augmentation
        x_i = augment(X, noise=0.2)
        x_j = augment(X, noise=0.2)
        z_i = x_i @ W
        z_j = x_j @ W
        loss = nt_xent_loss(z_i, z_j, temp)

        # numerical gradient (simple but slow)
        grad = np.zeros_like(W)
        eps = 1e-4
        for a in range(W.shape[0]):
            for b in range(W.shape[1]):
                W[a, b] += eps
                loss_p = nt_xent_loss((x_i @ W), (x_j @ W), temp)
                W[a, b] -= 2*eps
                loss_m = nt_xent_loss((x_i @ W), (x_j @ W), temp)
                W[a, b] += eps
                grad[a, b] = (loss_p - loss_m) / (2*eps)
        W -= lr * grad

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}  Loss={loss:.4f}")
    return W

# --- demo ---
np.random.seed(42)
# 3 clusters
X = np.vstack([np.random.randn(10, 4) + [3,0,0,0],
               np.random.randn(10, 4) + [0,3,0,0],
               np.random.randn(10, 4) + [0,0,3,0]])

W = train_encoder(X, d_proj=2, epochs=150, lr=0.05)
Z = normalize(X @ W)
# check: same-cluster similarity should be high
sims = Z @ Z.T
print(f"\nAvg within-cluster similarity:  {np.mean([sims[i,j] for i in range(10) for j in range(10) if i!=j]):.3f}")
print(f"Avg between-cluster similarity: {np.mean([sims[i,j] for i in range(10) for j in range(10,20)]):.3f}")
