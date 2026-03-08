"""Graph Neural Network — message passing on graph-structured data."""

import numpy as np

def normalize_adjacency(A):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    A_hat = A + np.eye(len(A))  # add self-loops
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

class GCN:
    def __init__(self, n_features, n_hidden, n_classes):
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2/n_features)
        self.W2 = np.random.randn(n_hidden, n_classes) * np.sqrt(2/n_hidden)

    def forward(self, A_norm, X):
        H = relu(A_norm @ X @ self.W1)
        return softmax(A_norm @ H @ self.W2)

    def train(self, A_norm, X, y_onehot, mask, lr=0.01, n_epochs=200):
        losses = []
        for _ in range(n_epochs):
            probs = self.forward(A_norm, X)
            loss = -np.mean(np.sum(y_onehot[mask] * np.log(probs[mask] + 1e-8), axis=1))
            losses.append(loss)
            # simplified gradient
            grad = probs.copy()
            grad[mask] -= y_onehot[mask]
            grad[~mask] = 0
            grad /= mask.sum()
            dW2 = relu(A_norm @ X @ self.W1).T @ (A_norm @ grad)
            self.W2 -= lr * dW2
        return losses

# --- demo ---
np.random.seed(42)

# Karate club-like graph (small)
n_nodes = 10
A = np.zeros((n_nodes, n_nodes))
edges = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(4,6),(5,6),(5,7),(6,7),(7,8),(7,9),(8,9)]
for i, j in edges:
    A[i,j] = A[j,i] = 1

A_norm = normalize_adjacency(A)
X = np.random.randn(n_nodes, 4)  # node features
labels = np.array([0,0,0,0,0,1,1,1,1,1])
y_onehot = np.eye(2)[labels]

# train on nodes 0,1,8,9 (semi-supervised)
train_mask = np.array([True,True,False,False,False,False,False,False,True,True])

gcn = GCN(4, 8, 2)
losses = gcn.train(A_norm, X, y_onehot, train_mask, lr=0.1, n_epochs=200)

preds = gcn.forward(A_norm, X).argmax(axis=1)
acc = (preds == labels).mean()

print("=== Graph Convolutional Network ===")
print(f"Nodes: {n_nodes}, Edges: {len(edges)}, Classes: 2")
print(f"Training nodes: {train_mask.sum()}, Test nodes: {(~train_mask).sum()}\n")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Overall accuracy: {acc:.0%}")
print(f"Predictions: {preds}")
print(f"True labels: {labels}")
