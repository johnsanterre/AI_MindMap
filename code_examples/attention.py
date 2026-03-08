"""Attention Mechanism — scaled dot-product self-attention."""

import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, Wq, Wk, Wv):
    """Single-head self-attention.
    X: (seq_len, d_model)
    Returns: (seq_len, d_v), attention_weights
    """
    Q = X @ Wq                          # queries
    K = X @ Wk                          # keys
    V = X @ Wv                          # values
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)   # scaled dot product
    weights = softmax(scores)            # attention weights
    output = weights @ V                 # weighted sum of values
    return output, weights

def multi_head_attention(X, heads):
    """Multi-head attention: concat multiple heads then project."""
    outputs = []
    all_weights = []
    for Wq, Wk, Wv in heads:
        out, w = self_attention(X, Wq, Wk, Wv)
        outputs.append(out)
        all_weights.append(w)
    return np.concatenate(outputs, axis=-1), all_weights

# --- demo ---
np.random.seed(42)
seq_len, d_model, d_k = 4, 8, 4

# simulate 4 token embeddings
X = np.random.randn(seq_len, d_model)
tokens = ["The", "cat", "sat", "down"]

# single head
Wq = np.random.randn(d_model, d_k) * 0.3
Wk = np.random.randn(d_model, d_k) * 0.3
Wv = np.random.randn(d_model, d_k) * 0.3

output, weights = self_attention(X, Wq, Wk, Wv)

print("Attention weights (who attends to whom):\n")
print(f"{'':>8}", end="")
for t in tokens:
    print(f"{t:>8}", end="")
print()
for i, t in enumerate(tokens):
    print(f"{t:>8}", end="")
    for j in range(seq_len):
        print(f"{weights[i,j]:8.3f}", end="")
    print()

print(f"\nOutput shape: {output.shape}")
