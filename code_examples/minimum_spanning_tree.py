"""Minimum Spanning Tree — Kruskal's algorithm with Union-Find."""

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, rank, a, b):
    a, b = find(parent, a), find(parent, b)
    if a == b:
        return False
    if rank[a] < rank[b]:
        a, b = b, a
    parent[b] = a
    if rank[a] == rank[b]:
        rank[a] += 1
    return True

def kruskal(vertices, edges):
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}
    mst, total = [], 0
    for w, u, v in sorted(edges):
        if union(parent, rank, u, v):
            mst.append((u, v, w))
            total += w
    return mst, total

# --- demo ---
vertices = ["A", "B", "C", "D", "E"]
edges = [
    (4, "A", "B"), (2, "A", "C"), (3, "B", "C"),
    (5, "B", "D"), (1, "C", "D"), (6, "D", "E"),
]
mst, cost = kruskal(vertices, edges)
print("MST edges:", mst)
print("Total cost:", cost)
