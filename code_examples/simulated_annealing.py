"""Simulated Annealing — probabilistic optimization inspired by metallurgy."""

import numpy as np

def simulated_annealing(f, x0, T0=10.0, T_min=0.001, alpha=0.995, n_iter=5000, step_size=0.5):
    x = x0.copy()
    best_x = x.copy()
    best_f = f(x)
    T = T0
    history = []

    for i in range(n_iter):
        # propose neighbor
        x_new = x + np.random.randn(len(x)) * step_size
        f_new = f(x_new)
        delta = f_new - f(x)

        # accept or reject
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x = x_new
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        T *= alpha
        if T < T_min:
            T = T_min
        history.append(best_f)

    return best_x, best_f, history

# --- demo ---
np.random.seed(42)

# Rastrigin function: many local minima
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# 2D Rastrigin: global minimum at (0, 0), value = 0
x0 = np.array([4.0, 4.0])

print("=== Simulated Annealing ===")
print(f"Rastrigin function (many local minima)")
print(f"Global minimum: f(0,0) = 0\n")

results = []
for trial in range(5):
    np.random.seed(trial)
    x_best, f_best, hist = simulated_annealing(rastrigin, x0, T0=20.0, alpha=0.998)
    results.append((f_best, x_best))
    print(f"  Trial {trial}: f={f_best:.4f} at ({x_best[0]:.4f}, {x_best[1]:.4f})")

best_trial = min(results, key=lambda r: r[0])
print(f"\nBest: f={best_trial[0]:.4f}")

# compare cooling schedules
print("\n--- Cooling Schedule Effect ---")
for alpha in [0.99, 0.995, 0.999]:
    np.random.seed(42)
    _, f_best, _ = simulated_annealing(rastrigin, x0, alpha=alpha, n_iter=5000)
    print(f"  α={alpha}: best f={f_best:.4f}")
