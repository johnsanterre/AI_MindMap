"""Random Walk — 1D, 2D, and properties."""

import numpy as np

def random_walk_1d(n_steps, n_walks=5):
    """Multiple 1D random walks."""
    steps = np.random.choice([-1, 1], size=(n_walks, n_steps))
    return np.cumsum(steps, axis=1)

def random_walk_2d(n_steps):
    """Single 2D random walk."""
    directions = np.random.randint(0, 4, n_steps)
    dx = np.where(directions == 0, 1, np.where(directions == 1, -1, 0))
    dy = np.where(directions == 2, 1, np.where(directions == 3, -1, 0))
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y

def return_probability(n_steps, n_sims=10000):
    """Estimate probability of returning to origin."""
    walks = random_walk_1d(n_steps, n_sims)
    returns = np.any(walks == 0, axis=1)
    return returns.mean()

# --- demo ---
np.random.seed(42)

# 1D walks
walks = random_walk_1d(100, 5)
print("=== 1D Random Walks (100 steps) ===")
for i in range(5):
    print(f"  Walk {i+1}: final position = {walks[i, -1]:+3d}, "
          f"max = {walks[i].max():+3d}, min = {walks[i].min():+3d}")

# theoretical: E[|X_n|] ≈ sqrt(2n/π)
n = 10000
big_walks = random_walk_1d(n, 1000)
empirical_dist = np.abs(big_walks[:, -1]).mean()
theoretical = np.sqrt(2 * n / np.pi)
print(f"\n  E[|X_{n}|]: empirical={empirical_dist:.1f}, theoretical={theoretical:.1f}")

# 2D walk
x, y = random_walk_2d(1000)
final_dist = np.sqrt(x[-1]**2 + y[-1]**2)
print(f"\n=== 2D Random Walk (1000 steps) ===")
print(f"  Final position: ({x[-1]}, {y[-1]}), distance={final_dist:.1f}")

# return probability
print(f"\n=== Return to Origin ===")
for n in [10, 50, 100, 500]:
    p = return_probability(n)
    print(f"  P(return in {n:3d} steps) = {p:.3f}")
