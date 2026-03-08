"""PageRank — Google's original ranking algorithm."""

import numpy as np

def pagerank(adj, damping=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank from adjacency dict {node: [neighbors]}."""
    nodes = sorted(adj.keys())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # build transition matrix
    M = np.zeros((n, n))
    for node in nodes:
        out = adj[node]
        if out:
            for target in out:
                M[idx[target], idx[node]] = 1.0 / len(out)
        else:  # dangling node
            M[:, idx[node]] = 1.0 / n

    # power iteration
    rank = np.ones(n) / n
    for iteration in range(max_iter):
        new_rank = damping * M @ rank + (1 - damping) / n
        if np.abs(new_rank - rank).max() < tol:
            break
        rank = new_rank

    return {nodes[i]: rank[i] for i in range(n)}

# --- demo ---
web = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"],
    "E": ["C", "D"],
    "F": ["B", "E"],
}

ranks = pagerank(web)
print("PageRank scores:")
for page, score in sorted(ranks.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 100)
    print(f"  {page}: {score:.4f}  {bar}")
