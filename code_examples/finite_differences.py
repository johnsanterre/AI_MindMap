"""Finite Differences — numerical differentiation and PDE solving."""

import numpy as np

def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2

def solve_heat_equation(nx=50, nt=500, L=1.0, T=0.1, alpha=1.0):
    """1D heat equation: u_t = alpha * u_xx."""
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    if r > 0.5:
        dt = 0.4 * dx**2 / alpha
        nt = int(T / dt)
        r = alpha * dt / dx**2

    x = np.linspace(0, L, nx)
    u = np.sin(np.pi * x)  # initial condition
    u[0] = u[-1] = 0  # boundary conditions

    for _ in range(nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u[0] = u[-1] = 0
        u = u_new

    exact = np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x)
    return x, u, exact

# --- demo ---
print("=== Finite Differences ===\n")

# numerical differentiation accuracy
f = np.sin
df_exact = np.cos
x0 = 1.0

print("--- Derivative of sin(x) at x=1 ---")
print(f"{'h':>12} {'Forward':>12} {'Central':>12} {'Fwd Error':>12} {'Ctr Error':>12}")
for h in [0.1, 0.01, 0.001, 1e-4, 1e-5]:
    fwd = forward_diff(f, x0, h)
    ctr = central_diff(f, x0, h)
    exact = df_exact(x0)
    print(f"{h:>12.0e} {fwd:>12.8f} {ctr:>12.8f} {abs(fwd-exact):>12.2e} {abs(ctr-exact):>12.2e}")

# heat equation
print("\n--- 1D Heat Equation ---")
x, u_numerical, u_exact = solve_heat_equation()
error = np.max(np.abs(u_numerical - u_exact))
print(f"Max error vs exact solution: {error:.6f}")
print(f"{'x':>6} {'Numerical':>12} {'Exact':>12}")
for i in range(0, len(x), 10):
    print(f"{x[i]:>6.2f} {u_numerical[i]:>12.6f} {u_exact[i]:>12.6f}")
