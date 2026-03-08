"""Shortest Path — Dijkstra's algorithm."""

import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def path_to(prev, target):
    path = []
    while target in prev:
        path.append(target)
        target = prev[target]
    path.append(target)
    return list(reversed(path))

# --- demo ---
graph = {
    "A": [("B", 2), ("C", 5)],
    "B": [("C", 1), ("D", 4)],
    "C": [("D", 2)],
    "D": [("E", 3)],
    "E": [],
}
dist, prev = dijkstra(graph, "A")
print("Distances from A:", dist)
print("Path A→E:", path_to(prev, "E"), f"(cost {dist['E']})")
