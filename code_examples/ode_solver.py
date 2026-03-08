"""ODE Solver — Euler and Runge-Kutta methods for differential equations."""

import numpy as np

def euler(f, y0, t_span, dt):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + dt * np.array(f(t[i], y[i]))
    return t, y

def rk4(f, y0, t_span, dt):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = np.array(f(t[i], y[i]))
        k2 = np.array(f(t[i] + dt/2, y[i] + dt/2 * k1))
        k3 = np.array(f(t[i] + dt/2, y[i] + dt/2 * k2))
        k4 = np.array(f(t[i] + dt, y[i] + dt * k3))
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# --- demo ---
# simple harmonic oscillator: y'' + y = 0 → [y, y'] = [y', -y]
def harmonic(t, state):
    y, v = state
    return [v, -y]

# exact solution: y(t) = cos(t)
exact = lambda t: np.cos(t)

print("=== ODE Solvers: Simple Harmonic Oscillator ===")
print("y'' + y = 0, y(0)=1, y'(0)=0\n")

for dt in [0.5, 0.1, 0.01]:
    t_e, y_e = euler(harmonic, [1.0, 0.0], [0, 10], dt)
    t_r, y_r = rk4(harmonic, [1.0, 0.0], [0, 10], dt)
    err_euler = np.max(np.abs(y_e[:, 0] - exact(t_e)))
    err_rk4 = np.max(np.abs(y_r[:, 0] - exact(t_r)))
    print(f"dt={dt:<5} Euler error: {err_euler:.6f}  RK4 error: {err_rk4:.2e}")

# Lorenz system demo
print("\n=== Lorenz Attractor ===")
def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

t, y = rk4(lorenz, [1, 1, 1], [0, 5], 0.01)
print(f"t=0: ({y[0,0]:.2f}, {y[0,1]:.2f}, {y[0,2]:.2f})")
print(f"t=5: ({y[-1,0]:.2f}, {y[-1,1]:.2f}, {y[-1,2]:.2f})")
print(f"Range: x=[{y[:,0].min():.1f}, {y[:,0].max():.1f}], z=[{y[:,2].min():.1f}, {y[:,2].max():.1f}]")
