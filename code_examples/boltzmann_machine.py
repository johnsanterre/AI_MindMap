"""Boltzmann Machine — Restricted Boltzmann Machine with contrastive divergence."""

import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b_v = np.zeros(n_visible)
        self.b_h = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def sample_hidden(self, v):
        p_h = self.sigmoid(v @ self.W + self.b_h)
        return p_h, (np.random.rand(*p_h.shape) < p_h).astype(float)

    def sample_visible(self, h):
        p_v = self.sigmoid(h @ self.W.T + self.b_v)
        return p_v, (np.random.rand(*p_v.shape) < p_v).astype(float)

    def contrastive_divergence(self, v, k=1, lr=0.01):
        """CD-k learning."""
        # positive phase
        p_h0, h0 = self.sample_hidden(v)
        pos_grad = v.T @ p_h0

        # negative phase (k steps of Gibbs sampling)
        h = h0
        for _ in range(k):
            p_v, v_sample = self.sample_visible(h)
            p_h, h = self.sample_hidden(v_sample)
        neg_grad = v_sample.T @ p_h

        # update
        batch_size = len(v)
        self.W += lr * (pos_grad - neg_grad) / batch_size
        self.b_v += lr * (v - v_sample).mean(axis=0)
        self.b_h += lr * (p_h0 - p_h).mean(axis=0)

        recon_error = np.mean((v - p_v)**2)
        return recon_error

    def free_energy(self, v):
        vbias = v @ self.b_v
        wx_b = v @ self.W + self.b_h
        hidden_term = np.sum(np.log(1 + np.exp(np.clip(wx_b, -10, 10))), axis=1)
        return -vbias - hidden_term

# --- demo ---
np.random.seed(42)

# binary patterns
patterns = np.array([
    [1,1,0,0,0,0], [1,1,1,0,0,0], [0,0,0,1,1,0], [0,0,0,1,1,1],
    [1,1,0,0,0,0], [1,1,1,0,0,0], [0,0,0,1,1,0], [0,0,0,1,1,1],
] * 10, dtype=float)

rbm = RBM(6, 4)
print("=== Restricted Boltzmann Machine ===")
print(f"Visible: 6, Hidden: 4\n")

print(f"{'Epoch':>6} {'Recon Error':>12}")
print("-" * 20)
for epoch in range(201):
    idx = np.random.permutation(len(patterns))
    err = rbm.contrastive_divergence(patterns[idx[:20]], k=1, lr=0.1)
    if epoch % 40 == 0:
        print(f"{epoch:>6} {err:>12.4f}")

# test reconstruction
print("\nReconstruction test:")
test = np.array([[1,1,0,0,0,0], [0,0,0,1,1,1], [1,0,1,0,1,0]])
for v in test:
    p_h, h = rbm.sample_hidden(v.reshape(1, -1))
    p_v, _ = rbm.sample_visible(p_h)
    print(f"  Input:  {v}  →  Recon: {p_v[0].round(2)}")
