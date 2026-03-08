"""Graph Coloring — greedy coloring algorithm."""

def greedy_color(graph):
    colors = {}
    for node in sorted(graph):
        used = {colors[nb] for nb in graph[node] if nb in colors}
        for c in range(len(graph)):
            if c not in used:
                colors[node] = c
                break
    return colors

def is_valid(graph, colors):
    return all(colors[u] != colors[v] for u in graph for v in graph[u] if v in colors)

# --- demo ---
# Petersen-like graph
graph = {
    0: [1, 4, 5], 1: [0, 2, 6], 2: [1, 3, 7],
    3: [2, 4, 8], 4: [0, 3, 9], 5: [0, 7, 8],
    6: [1, 8, 9], 7: [2, 5, 9], 8: [3, 5, 6],
    9: [4, 6, 7],
}
colors = greedy_color(graph)
names = ["red", "blue", "green", "yellow", "orange"]
print("Coloring:", {n: names[c] for n, c in sorted(colors.items())})
print("Chromatic number ≤", max(colors.values()) + 1)
print("Valid:", is_valid(graph, colors))
