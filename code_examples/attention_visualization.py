"""Attention Visualization — how attention weights distribute across tokens."""

import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights

def multi_head_attention(Q, K, V, n_heads=2):
    d = Q.shape[-1]
    head_dim = d // n_heads
    all_weights = []
    outputs = []
    for h in range(n_heads):
        s = h * head_dim
        e = s + head_dim
        out, w = scaled_dot_product_attention(Q[:, s:e], K[:, s:e], V[:, s:e])
        outputs.append(out)
        all_weights.append(w)
    return np.concatenate(outputs, axis=-1), all_weights

# --- demo ---
np.random.seed(42)

tokens = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(tokens)
d_model = 8

# random embeddings
embeddings = np.random.randn(n_tokens, d_model)

# self-attention
output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)

print("=== Attention Weight Visualization ===\n")
print("Self-attention weights:")
header = "        " + "".join(f"{t:>6}" for t in tokens)
print(header)
for i, tok in enumerate(tokens):
    row = f"{tok:>6}  " + "".join(f"{weights[i,j]:>6.3f}" for j in range(n_tokens))
    print(row)

# causal (autoregressive) attention
print("\nCausal attention weights:")
causal_mask = np.tril(np.ones((n_tokens, n_tokens), dtype=bool))
_, causal_weights = scaled_dot_product_attention(embeddings, embeddings, embeddings, mask=causal_mask)
print(header)
for i, tok in enumerate(tokens):
    row = f"{tok:>6}  " + "".join(f"{causal_weights[i,j]:>6.3f}" for j in range(n_tokens))
    print(row)

# multi-head
print(f"\nMulti-head attention (2 heads):")
_, head_weights = multi_head_attention(embeddings, embeddings, embeddings, n_heads=2)
for h, hw in enumerate(head_weights):
    print(f"  Head {h} top attention for each token:")
    for i, tok in enumerate(tokens):
        top = hw[i].argmax()
        print(f"    {tok} → {tokens[top]} ({hw[i,top]:.3f})")
