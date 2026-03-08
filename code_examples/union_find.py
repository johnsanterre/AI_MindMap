"""Union-Find — disjoint set with path compression and union by rank."""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        self.components -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

# --- demo ---
uf = UnionFind(7)
edges = [(0,1), (1,2), (3,4), (4,5), (5,6)]
for u, v in edges:
    uf.union(u, v)
    print(f"Union({u},{v}) → {uf.components} components")

print(f"\n0 and 2 connected? {uf.connected(0, 2)}")
print(f"0 and 3 connected? {uf.connected(0, 3)}")
uf.union(2, 3)
print(f"After Union(2,3): 0 and 5 connected? {uf.connected(0, 5)}")
