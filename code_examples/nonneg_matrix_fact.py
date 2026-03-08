"""Non-negative Matrix Factorization — parts-based decomposition."""

import numpy as np

def nmf(V, k, n_iter=200):
    """Multiplicative update rules for NMF: V ≈ W @ H."""
    n, m = V.shape
    W = np.random.rand(n, k) + 0.1
    H = np.random.rand(k, m) + 0.1

    for _ in range(n_iter):
        # update H
        H *= (W.T @ V) / (W.T @ W @ H + 1e-10)
        # update W
        W *= (V @ H.T) / (W @ H @ H.T + 1e-10)

    return W, H

# --- demo ---
np.random.seed(42)

# simulate document-term matrix (6 docs, 8 words)
# topic 1: words 0-3, topic 2: words 4-7
W_true = np.array([
    [3, 0], [3, 0], [2, 0],  # docs about topic 1
    [0, 3], [0, 2], [0, 3],  # docs about topic 2
], dtype=float)
H_true = np.array([
    [2, 3, 1, 2, 0, 0, 0, 0],  # topic 1 word probs
    [0, 0, 0, 0, 2, 1, 3, 2],  # topic 2 word probs
], dtype=float)
V = W_true @ H_true + np.random.rand(6, 8) * 0.5

W, H = nmf(V, k=2, n_iter=300)
V_approx = W @ H

print("=== Non-negative Matrix Factorization ===")
print(f"V ({V.shape[0]}×{V.shape[1]}) ≈ W ({W.shape[0]}×{W.shape[1]}) @ H ({H.shape[0]}×{H.shape[1]})\n")

print("Discovered topics (word weights):")
words = ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"]
for t in range(2):
    top_words = H[t].argsort()[-4:][::-1]
    print(f"  Topic {t}: {', '.join(f'{words[w]}({H[t,w]:.1f})' for w in top_words)}")

print(f"\nDocument-topic assignments:")
for d in range(6):
    topic = W[d].argmax()
    print(f"  Doc {d}: topic {topic} (weights: {W[d].round(2)})")

recon_err = np.linalg.norm(V - V_approx) / np.linalg.norm(V)
print(f"\nRelative reconstruction error: {recon_err:.4f}")
