"""Cosine Similarity — document similarity, embedding search, retrieval."""

import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-8)
    return X_norm @ X_norm.T

def top_k_similar(query_vec, corpus_vecs, k=3):
    sims = np.array([cosine_sim(query_vec, v) for v in corpus_vecs])
    top_idx = sims.argsort()[-k:][::-1]
    return [(i, sims[i]) for i in top_idx]

# --- demo ---
np.random.seed(42)

# simulate word embeddings (4D for clarity)
embeddings = {
    "king":    np.array([0.9, 0.1, 0.8, 0.2]),
    "queen":   np.array([0.8, 0.2, 0.7, 0.9]),
    "man":     np.array([0.7, 0.1, 0.5, 0.1]),
    "woman":   np.array([0.6, 0.2, 0.4, 0.8]),
    "prince":  np.array([0.85, 0.15, 0.7, 0.15]),
    "car":     np.array([0.1, 0.9, 0.1, 0.3]),
    "truck":   np.array([0.15, 0.85, 0.15, 0.35]),
    "bicycle": np.array([0.1, 0.7, 0.05, 0.2]),
}

print("=== Pairwise Cosine Similarities ===")
words = list(embeddings.keys())
for i in range(len(words)):
    for j in range(i+1, len(words)):
        sim = cosine_sim(embeddings[words[i]], embeddings[words[j]])
        if sim > 0.9:
            print(f"  {words[i]:>8} ↔ {words[j]:<8} = {sim:.3f} ★")
        elif sim > 0.7:
            print(f"  {words[i]:>8} ↔ {words[j]:<8} = {sim:.3f}")

# analogy: king - man + woman ≈ queen
print("\n=== Word Analogy: king - man + woman = ? ===")
query = embeddings["king"] - embeddings["man"] + embeddings["woman"]
vecs = np.array(list(embeddings.values()))
results = top_k_similar(query, vecs, k=3)
for idx, sim in results:
    print(f"  {words[idx]:>8}: {sim:.3f}")
