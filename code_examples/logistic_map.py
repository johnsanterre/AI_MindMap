"""Logistic Map — chaos, bifurcation, and sensitivity to initial conditions."""

import numpy as np

def logistic_map(r, x0, n_steps):
    x = np.zeros(n_steps)
    x[0] = x0
    for t in range(1, n_steps):
        x[t] = r * x[t-1] * (1 - x[t-1])
    return x

def lyapunov_exponent(r, x0=0.1, n_steps=1000, n_discard=100):
    x = x0
    lyap = 0
    for t in range(n_steps):
        x = r * x * (1 - x)
        if t >= n_discard:
            deriv = abs(r * (1 - 2*x))
            if deriv > 0:
                lyap += np.log(deriv)
    return lyap / (n_steps - n_discard)

# --- demo ---
print("=== Logistic Map: x_{n+1} = r·x_n·(1 - x_n) ===\n")

# different regimes
print("--- Behavior at different r values ---")
for r in [2.0, 3.2, 3.5, 3.9]:
    orbit = logistic_map(r, 0.5, 200)
    tail = orbit[-20:]
    unique = np.unique(tail.round(4))
    lyap = lyapunov_exponent(r)
    if len(unique) <= 4:
        print(f"r={r}: period-{len(unique)}, attractors={unique.round(4)}, λ={lyap:.3f}")
    else:
        print(f"r={r}: chaotic, range=[{tail.min():.3f}, {tail.max():.3f}], λ={lyap:.3f}")

# sensitivity to initial conditions
print("\n--- Sensitivity to Initial Conditions (r=3.9) ---")
x1 = logistic_map(3.9, 0.5, 30)
x2 = logistic_map(3.9, 0.5 + 1e-10, 30)
print(f"{'Step':>5} {'x(0.5)':>10} {'x(0.5+ε)':>10} {'Diff':>12}")
for t in [0, 5, 10, 15, 20, 25]:
    print(f"{t:>5} {x1[t]:>10.6f} {x2[t]:>10.6f} {abs(x1[t]-x2[t]):>12.2e}")

# bifurcation data
print("\n--- Bifurcation Points ---")
print(f"r=1: single fixed point (0)")
print(f"r=3: period-2 bifurcation")
print(f"r≈3.449: period-4")
print(f"r≈3.570: onset of chaos")
print(f"Feigenbaum constant δ ≈ 4.669 (universal ratio of bifurcation intervals)")
