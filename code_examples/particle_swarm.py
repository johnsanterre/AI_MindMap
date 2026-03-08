"""Particle Swarm Optimization — swarm intelligence for global optimization."""

import numpy as np

def pso(f, n_dims, n_particles=30, n_iter=200, bounds=(-5, 5),
        w=0.7, c1=1.5, c2=1.5):
    """Particle Swarm Optimization."""
    # initialize
    pos = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dims))
    vel = np.random.randn(n_particles, n_dims) * 0.1
    p_best = pos.copy()
    p_best_val = np.array([f(p) for p in pos])
    g_best = p_best[p_best_val.argmin()].copy()
    g_best_val = p_best_val.min()
    history = [g_best_val]

    for _ in range(n_iter):
        r1 = np.random.rand(n_particles, n_dims)
        r2 = np.random.rand(n_particles, n_dims)
        vel = w * vel + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos)
        pos = np.clip(pos + vel, bounds[0], bounds[1])

        vals = np.array([f(p) for p in pos])
        improved = vals < p_best_val
        p_best[improved] = pos[improved]
        p_best_val[improved] = vals[improved]

        if vals.min() < g_best_val:
            g_best = pos[vals.argmin()].copy()
            g_best_val = vals.min()
        history.append(g_best_val)

    return g_best, g_best_val, history

# --- demo ---
np.random.seed(42)

# Ackley function: many local minima, global min at origin
def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) \
           - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

print("=== Particle Swarm Optimization ===")
print(f"Ackley function (5D), global min at origin = 0\n")

best, best_val, hist = pso(ackley, n_dims=5, n_particles=40, n_iter=300)
print(f"Best: f = {best_val:.6f}")
print(f"At: {best.round(4)}")
print(f"\nConvergence:")
for i in [0, 10, 50, 100, 200, 300]:
    print(f"  Iter {i:>3}: f = {hist[i]:.6f}")
