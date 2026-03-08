"""Topic Modeling — Latent Dirichlet Allocation via collapsed Gibbs sampling."""

import numpy as np

def lda_gibbs(docs, V, K, n_iter=100, alpha=0.1, beta=0.01):
    """Collapsed Gibbs sampling for LDA."""
    # initialize
    n_docs = len(docs)
    z = []  # topic assignments
    n_dk = np.zeros((n_docs, K))  # doc-topic counts
    n_kw = np.zeros((K, V))       # topic-word counts
    n_k = np.zeros(K)             # topic counts

    for d, doc in enumerate(docs):
        z_d = []
        for w in doc:
            t = np.random.randint(K)
            z_d.append(t)
            n_dk[d, t] += 1
            n_kw[t, w] += 1
            n_k[t] += 1
        z.append(z_d)

    # Gibbs sampling
    for _ in range(n_iter):
        for d, doc in enumerate(docs):
            for i, w in enumerate(doc):
                t = z[d][i]
                n_dk[d, t] -= 1
                n_kw[t, w] -= 1
                n_k[t] -= 1

                # conditional distribution
                p = (n_dk[d] + alpha) * (n_kw[:, w] + beta) / (n_k + V * beta)
                p /= p.sum()
                t_new = np.random.choice(K, p=p)

                z[d][i] = t_new
                n_dk[d, t_new] += 1
                n_kw[t_new, w] += 1
                n_k[t_new] += 1

    # compute distributions
    theta = (n_dk + alpha) / (n_dk + alpha).sum(axis=1, keepdims=True)
    phi = (n_kw + beta) / (n_kw + beta).sum(axis=1, keepdims=True)
    return theta, phi

# --- demo ---
np.random.seed(42)

vocab = ["neural", "network", "gradient", "loss",
         "matrix", "vector", "eigenvalue", "linear",
         "probability", "bayes", "distribution", "sample"]
V = len(vocab)
K = 3  # topics

# synthetic docs (ML, linear algebra, statistics)
doc_topics = [
    [0,0,1,2,3,0,1],   # ML-heavy
    [4,5,6,7,4,5],      # linear algebra
    [8,9,10,11,8,9],    # statistics
    [0,1,4,5,2,3],      # ML + linalg
    [8,9,0,1,10,11],    # stats + ML
    [4,5,6,7,6,7],      # linalg
]

theta, phi = lda_gibbs(doc_topics, V, K, n_iter=200)

print("=== LDA Topic Modeling ===")
print(f"Docs: {len(doc_topics)}, Vocab: {V}, Topics: {K}\n")

print("Discovered topics (top words):")
for k in range(K):
    top = phi[k].argsort()[-4:][::-1]
    words = ", ".join(f"{vocab[w]}({phi[k,w]:.2f})" for w in top)
    print(f"  Topic {k}: {words}")

print(f"\nDocument-topic distributions:")
for d in range(len(doc_topics)):
    print(f"  Doc {d}: {theta[d].round(2)}")
