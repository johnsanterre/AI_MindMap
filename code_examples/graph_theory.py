"""Graph Theory — adjacency list, BFS, DFS."""

def make_graph(edges, directed=False):
    g = {}
    for u, v in edges:
        g.setdefault(u, []).append(v)
        if not directed:
            g.setdefault(v, []).append(u)
    return g

def bfs(graph, start):
    visited, queue, order = {start}, [start], []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nb in graph.get(node, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order

def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for nb in graph.get(start, []):
        if nb not in visited:
            dfs(graph, nb, visited)
    return visited

# --- demo ---
edges = [("A","B"), ("A","C"), ("B","D"), ("C","D"), ("D","E")]
G = make_graph(edges)
print("Graph:", {k: sorted(v) for k, v in sorted(G.items())})
print("BFS from A:", bfs(G, "A"))
print("DFS from A:", dfs(G, "A"))
