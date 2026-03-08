"""TF-IDF — term frequency–inverse document frequency from scratch."""

import numpy as np
from collections import Counter
import math

def compute_tf(doc):
    words = doc.lower().split()
    counts = Counter(words)
    total = len(words)
    return {w: c / total for w, c in counts.items()}

def compute_idf(docs):
    n = len(docs)
    doc_freq = Counter()
    for doc in docs:
        words = set(doc.lower().split())
        for w in words:
            doc_freq[w] += 1
    return {w: math.log(n / df) for w, df in doc_freq.items()}

def tfidf(docs):
    idf = compute_idf(docs)
    vocab = sorted(idf.keys())
    matrix = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        tf = compute_tf(doc)
        for j, word in enumerate(vocab):
            matrix[i, j] = tf.get(word, 0) * idf[word]
    return matrix, vocab

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# --- demo ---
docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "machine learning is fun",
    "deep learning uses neural networks",
]

matrix, vocab = tfidf(docs)
print("TF-IDF matrix shape:", matrix.shape)
print(f"Vocabulary size: {len(vocab)}\n")

print("Top TF-IDF terms per document:")
for i, doc in enumerate(docs):
    top_idx = matrix[i].argsort()[-3:][::-1]
    terms = [(vocab[j], matrix[i, j]) for j in top_idx if matrix[i, j] > 0]
    print(f"  Doc {i}: \"{doc[:30]}...\"")
    print(f"         {[(t, f'{s:.3f}') for t, s in terms]}")

print("\nDocument similarities:")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = cosine_sim(matrix[i], matrix[j])
        if sim > 0.01:
            print(f"  Doc {i} ↔ Doc {j}: {sim:.3f}")
