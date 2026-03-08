"""Triplet Loss — learning embeddings via anchor-positive-negative triplets."""

import numpy as np

def relu(x): return np.maximum(0, x)

def embed(X, W1, W2):
    h = relu(X @ W1)
    e = h @ W2
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)

def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = np.sum((anchor - positive)**2, axis=-1)
    d_neg = np.sum((anchor - negative)**2, axis=-1)
    return np.maximum(0, d_pos - d_neg + margin).mean()

def mine_hard_negatives(embeddings, labels, anchor_idx):
    """Find hardest negative: closest point from different class."""
    anchor_label = labels[anchor_idx]
    anchor_emb = embeddings[anchor_idx]
    neg_mask = labels != anchor_label
    if not neg_mask.any():
        return None
    dists = np.sum((embeddings[neg_mask] - anchor_emb)**2, axis=1)
    return np.where(neg_mask)[0][dists.argmin()]

# --- demo ---
np.random.seed(42)
d_in, d_hidden, d_embed = 6, 8, 3
n_per_class = 15

# 4 classes
data = []
labels = []
for c in range(4):
    center = np.random.randn(d_in) * 2
    data.append(center + np.random.randn(n_per_class, d_in) * 0.5)
    labels.extend([c] * n_per_class)
X = np.vstack(data)
labels = np.array(labels)

W1 = np.random.randn(d_in, d_hidden) * np.sqrt(2/d_in)
W2 = np.random.randn(d_hidden, d_embed) * np.sqrt(2/d_hidden)
embeddings = embed(X, W1, W2)

# generate triplets
triplets = []
for i in range(len(X)):
    pos_mask = (labels == labels[i]) & (np.arange(len(X)) != i)
    neg_mask = labels != labels[i]
    if pos_mask.any() and neg_mask.any():
        pos = np.random.choice(np.where(pos_mask)[0])
        neg = mine_hard_negatives(embeddings, labels, i)
        if neg is not None:
            triplets.append((i, pos, neg))

anchors = embeddings[[t[0] for t in triplets[:20]]]
positives = embeddings[[t[1] for t in triplets[:20]]]
negatives = embeddings[[t[2] for t in triplets[:20]]]

loss = triplet_loss(anchors, positives, negatives, margin=0.5)
d_pos = np.mean(np.sum((anchors - positives)**2, axis=1))
d_neg = np.mean(np.sum((anchors - negatives)**2, axis=1))

print("=== Triplet Loss ===")
print(f"Classes: 4, Samples/class: {n_per_class}, Embedding dim: {d_embed}")
print(f"Triplets: {len(triplets)}\n")
print(f"Triplet loss (margin=0.5): {loss:.4f}")
print(f"Avg pos distance: {d_pos:.4f}")
print(f"Avg neg distance: {d_neg:.4f}")
print(f"Margin satisfied: {(d_neg - d_pos > 0.5).mean():.0%} of triplets")
print(f"\nHard negative mining selects the closest wrong-class example.")
print(f"Used in: FaceNet, image retrieval, metric learning.")
