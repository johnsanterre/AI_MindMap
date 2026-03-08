"""Transformer — single-layer transformer encoder block."""

import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def multi_head_attention(Q, K, V, n_heads):
    seq_len, d = Q.shape
    d_k = d // n_heads
    out = np.zeros_like(Q)
    for h in range(n_heads):
        s = slice(h*d_k, (h+1)*d_k)
        scores = Q[:, s] @ K[:, s].T / np.sqrt(d_k)
        weights = softmax(scores)
        out[:, s] = weights @ V[:, s]
    return out

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def transformer_block(x, Wq, Wk, Wv, Wo, W1, b1, W2, b2, n_heads=2):
    # self-attention + residual + norm
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    attn = multi_head_attention(Q, K, V, n_heads)
    x = layer_norm(x + attn @ Wo)
    # feed-forward + residual + norm
    ff = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff)
    return x

# --- demo ---
np.random.seed(42)
seq_len, d_model, d_ff, n_heads = 6, 8, 16, 2
s = 0.3

Wq = np.random.randn(d_model, d_model) * s
Wk = np.random.randn(d_model, d_model) * s
Wv = np.random.randn(d_model, d_model) * s
Wo = np.random.randn(d_model, d_model) * s
W1 = np.random.randn(d_model, d_ff) * s
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * s
b2 = np.zeros(d_model)

x = np.random.randn(seq_len, d_model)
tokens = ["The", "cat", "sat", "on", "the", "mat"]

out = transformer_block(x, Wq, Wk, Wv, Wo, W1, b1, W2, b2, n_heads)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"\nToken representations (first 4 dims):")
for t, o in zip(tokens, out):
    print(f"  {t:>4}: {o[:4].round(3)}")
