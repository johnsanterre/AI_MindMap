"""Greedy Algorithms — activity selection, Huffman coding, fractional knapsack."""

def activity_selection(activities):
    """Select max non-overlapping activities. Each is (start, end)."""
    by_end = sorted(activities, key=lambda x: x[1])
    selected = [by_end[0]]
    for s, e in by_end[1:]:
        if s >= selected[-1][1]:
            selected.append((s, e))
    return selected

def huffman(freq):
    """Build Huffman tree, return code table."""
    import heapq
    heap = [(f, i, c) for i, (c, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    nodes = {}
    idx = len(freq)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        nodes[idx] = (n1, n2)
        heapq.heappush(heap, (f1 + f2, idx, idx))
        idx += 1
    codes = {}
    def walk(node, prefix=""):
        if node in nodes:
            walk(nodes[node][0], prefix + "0")
            walk(nodes[node][1], prefix + "1")
        else:
            codes[node] = prefix
    walk(heap[0][2])
    return codes

def fractional_knapsack(items, capacity):
    """items: list of (value, weight). Returns max value."""
    by_ratio = sorted(items, key=lambda x: x[0]/x[1], reverse=True)
    total = 0
    for v, w in by_ratio:
        take = min(w, capacity)
        total += v * (take / w)
        capacity -= take
        if capacity <= 0:
            break
    return total

# --- demo ---
acts = [(1,3), (2,5), (3,6), (5,7), (6,9), (8,10)]
print("Activities:", activity_selection(acts))

freq = {"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5}
codes = huffman(freq)
print("Huffman codes:", {chr(c) if isinstance(c, int) else c: v for c, v in sorted(codes.items(), key=lambda x: len(x[1]))})

items = [(60, 10), (100, 20), (120, 30)]
print("Fractional knapsack (cap=50):", fractional_knapsack(items, 50))
