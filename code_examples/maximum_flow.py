"""Maximum Flow — Edmonds-Karp (BFS-based Ford-Fulkerson)."""

from collections import deque

def bfs_path(graph, source, sink):
    visited = {source}
    queue = deque([(source, [source])])
    while queue:
        node, path = queue.popleft()
        for nb in graph.get(node, {}):
            if nb not in visited and graph[node][nb] > 0:
                visited.add(nb)
                if nb == sink:
                    return path + [nb]
                queue.append((nb, path + [nb]))
    return None

def max_flow(cap, source, sink):
    # build residual graph
    graph = {}
    for u in cap:
        for v in cap[u]:
            graph.setdefault(u, {})[v] = cap[u][v]
            graph.setdefault(v, {}).setdefault(u, 0)

    flow = 0
    while True:
        path = bfs_path(graph, source, sink)
        if not path:
            break
        # find bottleneck
        bn = min(graph[path[i]][path[i+1]] for i in range(len(path)-1))
        for i in range(len(path)-1):
            graph[path[i]][path[i+1]] -= bn
            graph[path[i+1]][path[i]] += bn
        flow += bn
    return flow

# --- demo ---
capacity = {
    "S": {"A": 10, "B": 8},
    "A": {"C": 5, "D": 7},
    "B": {"A": 3, "D": 6},
    "C": {"T": 10},
    "D": {"T": 8},
}
print("Max flow S→T:", max_flow(capacity, "S", "T"))
