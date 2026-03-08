"""Heaps and Priority Queues — min-heap from scratch."""

class MinHeap:
    def __init__(self):
        self.data = []

    def push(self, val):
        self.data.append(val)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        if len(self.data) == 1:
            return self.data.pop()
        top = self.data[0]
        self.data[0] = self.data.pop()
        self._sift_down(0)
        return top

    def peek(self):
        return self.data[0]

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.data[i] < self.data[parent]:
                self.data[i], self.data[parent] = self.data[parent], self.data[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            smallest = i
            left, right = 2*i+1, 2*i+2
            if left < n and self.data[left] < self.data[smallest]:
                smallest = left
            if right < n and self.data[right] < self.data[smallest]:
                smallest = right
            if smallest == i:
                break
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest

    def __len__(self):
        return len(self.data)

# --- demo ---
h = MinHeap()
for x in [7, 3, 9, 1, 5, 8, 2]:
    h.push(x)
    print(f"Push {x} → heap: {h.data}")

print("\nPop in order:")
while h:
    print(f"  {h.pop()}")
