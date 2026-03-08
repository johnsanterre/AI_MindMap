"""Word2Vec — skip-gram with negative sampling (minimal version)."""

import numpy as np
from collections import Counter

def build_vocab(corpus, min_count=1):
    words = corpus.split()
    counts = Counter(words)
    vocab = {w: i for i, (w, c) in enumerate(counts.most_common()) if c >= min_count}
    return vocab, words

def get_training_pairs(words, vocab, window=2):
    pairs = []
    for i, w in enumerate(words):
        if w not in vocab:
            continue
        for j in range(max(0, i-window), min(len(words), i+window+1)):
            if i != j and words[j] in vocab:
                pairs.append((vocab[w], vocab[words[j]]))
    return pairs

def train_skipgram(pairs, vocab_size, dim=10, lr=0.025, epochs=50, neg=5):
    W = np.random.randn(vocab_size, dim) * 0.1  # target embeddings
    C = np.random.randn(vocab_size, dim) * 0.1  # context embeddings

    for epoch in range(epochs):
        loss = 0
        np.random.shuffle(pairs)
        for target, context in pairs:
            # positive sample
            score = W[target] @ C[context]
            sig = 1 / (1 + np.exp(-np.clip(score, -10, 10)))
            grad = (sig - 1)
            W[target] -= lr * grad * C[context]
            C[context] -= lr * grad * W[target]
            loss -= np.log(sig + 1e-8)

            # negative samples
            for _ in range(neg):
                neg_idx = np.random.randint(vocab_size)
                if neg_idx == context:
                    continue
                score = W[target] @ C[neg_idx]
                sig = 1 / (1 + np.exp(-np.clip(score, -10, 10)))
                W[target] -= lr * sig * C[neg_idx]
                C[neg_idx] -= lr * sig * W[target]
                loss -= np.log(1 - sig + 1e-8)

    return W

def most_similar(word, vocab, W, top=3):
    if word not in vocab:
        return []
    idx = vocab[word]
    sims = W @ W[idx] / (np.linalg.norm(W, axis=1) * np.linalg.norm(W[idx]) + 1e-8)
    sims[idx] = -1
    inv = {v: k for k, v in vocab.items()}
    return [(inv[i], sims[i]) for i in sims.argsort()[-top:][::-1]]

# --- demo ---
np.random.seed(42)
corpus = ("the king sat on the throne . the queen sat beside the king . "
          "a man walked to the castle . a woman walked to the village . "
          "the king rules the kingdom . the queen rules with the king . "
          "the man works in the village . the woman works in the castle .")

vocab, words = build_vocab(corpus)
pairs = get_training_pairs(words, vocab, window=2)
W = train_skipgram(pairs, len(vocab), dim=16, epochs=100)

print(f"Vocab size: {len(vocab)}, Training pairs: {len(pairs)}")
for word in ["king", "queen", "man", "castle"]:
    sims = most_similar(word, vocab, W, top=3)
    print(f"  {word:>8} → {[(w, f'{s:.2f}') for w, s in sims]}")
