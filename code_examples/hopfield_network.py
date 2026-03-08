"""Hopfield Network — associative memory via energy minimization."""

import numpy as np

class HopfieldNetwork:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train(self, patterns):
        """Hebbian learning: W = sum(p @ p.T) / N."""
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= len(patterns)
        np.fill_diagonal(self.W, 0)

    def recall(self, x, n_steps=20):
        """Asynchronous update until convergence."""
        state = x.copy()
        for _ in range(n_steps):
            prev = state.copy()
            for i in np.random.permutation(self.n):
                state[i] = 1 if self.W[i] @ state >= 0 else -1
            if np.array_equal(state, prev):
                break
        return state

    def energy(self, x):
        return -0.5 * x @ self.W @ x

# --- demo ---
np.random.seed(42)

# store 3 patterns (8-bit)
patterns = np.array([
    [ 1, 1, 1,-1,-1,-1, 1, 1],
    [-1,-1, 1, 1, 1, 1,-1,-1],
    [ 1,-1, 1,-1, 1,-1, 1,-1],
])

net = HopfieldNetwork(8)
net.train(patterns)

print("=== Hopfield Network: Associative Memory ===")
print(f"Stored {len(patterns)} patterns of length {net.n}\n")

print("Pattern recall from noisy inputs:")
for i, p in enumerate(patterns):
    # flip 2 random bits
    noisy = p.copy()
    flip = np.random.choice(8, 2, replace=False)
    noisy[flip] *= -1
    recalled = net.recall(noisy)
    match = np.array_equal(recalled, p)
    n_errors = (noisy != p).sum()
    print(f"  Pattern {i}: {p}")
    print(f"  Noisy({n_errors}):   {noisy}")
    print(f"  Recalled:  {recalled}  {'✓ correct' if match else '✗ wrong'}\n")

# capacity
print("Capacity test:")
for n_patterns in [1, 3, 5, 8]:
    pats = np.sign(np.random.randn(n_patterns, 8))
    net.train(pats)
    correct = sum(np.array_equal(net.recall(p), p) for p in pats)
    print(f"  {n_patterns} patterns stored: {correct}/{n_patterns} recalled correctly")
print(f"\nTheoretical capacity: ~{int(0.138 * 8)} patterns (0.138N)")
