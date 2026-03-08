"""Embedding Layer — learned dense representations for discrete tokens."""

import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.1
        self.vocab_size = vocab_size

    def forward(self, indices):
        """Lookup: indices → dense vectors."""
        return self.W[indices]

    def backward(self, indices, grad_output, lr=0.01):
        """Update only the rows that were accessed."""
        for i, idx in enumerate(indices.flatten()):
            self.W[idx] -= lr * grad_output.reshape(-1, self.W.shape[1])[i]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# --- demo ---
np.random.seed(42)

vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park", "<pad>"]
vocab_size = len(vocab)
embed_dim = 8

emb = Embedding(vocab_size, embed_dim)

# sentence: "the cat sat on the mat"
sentence = np.array([0, 1, 2, 3, 0, 4])
vectors = emb.forward(sentence)

print("=== Embedding Layer ===")
print(f"Vocab: {vocab_size} words, Embed dim: {embed_dim}\n")

print("Sentence: 'the cat sat on the mat'")
print(f"Token IDs: {sentence}")
print(f"Embedding shape: {vectors.shape}\n")

# same token gets same embedding
print(f"'the' appears at positions 0 and 4:")
print(f"  Same vector? {np.array_equal(vectors[0], vectors[4])}")

# initial similarities (random, but structure emerges with training)
print(f"\nInitial cosine similarities:")
for i in range(min(5, vocab_size)):
    for j in range(i+1, min(5, vocab_size)):
        sim = cosine_sim(emb.W[i], emb.W[j])
        if abs(sim) > 0.3:
            print(f"  {vocab[i]:>5} ↔ {vocab[j]:<5}: {sim:.3f}")

# parameter count comparison
seq_len = 512
print(f"\n--- Parameter Efficiency ---")
print(f"One-hot ({vocab_size}×{seq_len}):     {vocab_size * seq_len} values")
print(f"Embedding ({vocab_size}×{embed_dim}): {vocab_size * embed_dim} params")
print(f"Embeddings are {vocab_size*seq_len/(vocab_size*embed_dim):.0f}× more compact")
