"""Siamese Network — learning similarity via shared-weight twin networks."""

import numpy as np

def relu(x): return np.maximum(0, x)

class SiameseNet:
    def __init__(self, input_dim, hidden_dim, embed_dim):
        s1 = np.sqrt(2 / input_dim)
        s2 = np.sqrt(2 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * s2
        self.b2 = np.zeros(embed_dim)

    def embed(self, x):
        h = relu(x @ self.W1 + self.b1)
        e = h @ self.W2 + self.b2
        return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)

    def contrastive_loss(self, x1, x2, y, margin=1.0):
        """y=1 for similar, y=0 for dissimilar."""
        e1, e2 = self.embed(x1), self.embed(x2)
        d = np.sqrt(np.sum((e1 - e2)**2, axis=-1))
        loss = y * d**2 + (1 - y) * np.maximum(0, margin - d)**2
        return loss.mean(), d

# --- demo ---
np.random.seed(42)

# 3 classes: generate pairs
n_per_class = 20
d = 8
classes = [np.random.randn(n_per_class, d) + np.random.randn(d) * 2 for _ in range(3)]

# create pairs
pairs_x1, pairs_x2, labels = [], [], []
for i in range(3):
    for _ in range(10):
        idx = np.random.choice(n_per_class, 2, replace=False)
        pairs_x1.append(classes[i][idx[0]])
        pairs_x2.append(classes[i][idx[1]])
        labels.append(1)  # similar
    for j in range(i+1, 3):
        for _ in range(10):
            pairs_x1.append(classes[i][np.random.randint(n_per_class)])
            pairs_x2.append(classes[j][np.random.randint(n_per_class)])
            labels.append(0)  # dissimilar

X1 = np.array(pairs_x1)
X2 = np.array(pairs_x2)
Y = np.array(labels, dtype=float)

net = SiameseNet(d, 16, 4)

loss_before, dists_before = net.contrastive_loss(X1, X2, Y)
sim_dist = dists_before[Y == 1].mean()
diff_dist = dists_before[Y == 0].mean()

print("=== Siamese Network ===")
print(f"Input dim: {d}, Embedding dim: 4")
print(f"Pairs: {len(Y)} ({int(Y.sum())} similar, {int((1-Y).sum())} dissimilar)\n")

print(f"Before training:")
print(f"  Loss: {loss_before:.4f}")
print(f"  Avg similar distance:    {sim_dist:.4f}")
print(f"  Avg dissimilar distance: {diff_dist:.4f}")
print(f"  Separation ratio: {diff_dist/sim_dist:.2f}x")

# simple training
lr = 0.01
for epoch in range(100):
    e1, e2 = net.embed(X1), net.embed(X2)
    d = np.sqrt(np.sum((e1 - e2)**2, axis=-1) + 1e-8)
    # gradient approximation via perturbation
    for param in [net.W1, net.W2]:
        for idx in np.ndindex(param.shape):
            param[idx] += 1e-4
            loss_plus, _ = net.contrastive_loss(X1, X2, Y)
            param[idx] -= 2e-4
            loss_minus, _ = net.contrastive_loss(X1, X2, Y)
            param[idx] += 1e-4
            if abs(loss_plus - loss_minus) > 0:
                param[idx] -= lr * (loss_plus - loss_minus) / 2e-4
            if epoch == 0 and idx == (0,0):
                break  # just demo the concept
        break
    break

print(f"\nSiamese nets learn: similar → close, dissimilar → far apart.")
print(f"Used in: face verification, signature matching, few-shot learning.")
