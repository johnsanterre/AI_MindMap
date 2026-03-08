"""Hashing — hash table with chaining."""

class HashTable:
    def __init__(self, size=8):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))
        self.count += 1

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def load_factor(self):
        return self.count / self.size

# --- demo ---
ht = HashTable()
data = {"apple": 3, "banana": 5, "cherry": 2, "date": 8, "elderberry": 1}
for k, v in data.items():
    ht.put(k, v)

print("Get 'cherry':", ht.get("cherry"))
print("Load factor:", ht.load_factor())
print("Buckets:")
for i, b in enumerate(ht.buckets):
    if b:
        print(f"  [{i}] {b}")
