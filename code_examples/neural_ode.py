"""Neural ODE — continuous-depth neural networks as differential equations."""

import numpy as np

def relu(x): return np.maximum(0, x)

class NeuralODEFunc:
    """f(t, z) = W2 * relu(W1 * z + b1) + b2."""
    def __init__(self, d):
        s = np.sqrt(2/d)
        self.W1 = np.random.randn(d, d) * s
        self.b1 = np.zeros(d)
        self.W2 = np.random.randn(d, d) * s
        self.b2 = np.zeros(d)

    def __call__(self, t, z):
        return self.W2 @ relu(self.W1 @ z + self.b1) + self.b2

def euler_integrate(func, z0, t_span, n_steps=50):
    """dz/dt = f(t, z), integrate from t0 to t1."""
    t = np.linspace(t_span[0], t_span[1], n_steps)
    dt = t[1] - t[0]
    z = z0.copy()
    trajectory = [z.copy()]
    for ti in t[:-1]:
        z = z + dt * func(ti, z)
        trajectory.append(z.copy())
    return np.array(trajectory), t

def rk4_integrate(func, z0, t_span, n_steps=50):
    t = np.linspace(t_span[0], t_span[1], n_steps)
    dt = t[1] - t[0]
    z = z0.copy()
    trajectory = [z.copy()]
    for ti in t[:-1]:
        k1 = func(ti, z)
        k2 = func(ti + dt/2, z + dt/2 * k1)
        k3 = func(ti + dt/2, z + dt/2 * k2)
        k4 = func(ti + dt, z + dt * k3)
        z = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(z.copy())
    return np.array(trajectory), t

# --- demo ---
np.random.seed(42)

d = 4
func = NeuralODEFunc(d)
z0 = np.array([1.0, 0.5, -0.3, 0.8])

traj_euler, t = euler_integrate(func, z0, (0, 1), n_steps=100)
traj_rk4, _ = rk4_integrate(func, z0, (0, 1), n_steps=100)

print("=== Neural ODE ===")
print(f"dz/dt = f_θ(t, z), z ∈ R^{d}\n")

print(f"{'t':>5} {'||z|| Euler':>14} {'||z|| RK4':>14}")
print("-" * 36)
for i in [0, 25, 50, 75, 99]:
    print(f"{t[i]:>5.2f} {np.linalg.norm(traj_euler[i]):>14.4f} {np.linalg.norm(traj_rk4[i]):>14.4f}")

# ResNet as discrete ODE
print(f"\nNeural ODE vs ResNet:")
print(f"  ResNet:     z_{'{t+1}'} = z_t + f(z_t)       (Euler, dt=1)")
print(f"  Neural ODE: dz/dt = f(t, z)          (continuous)")
print(f"  Benefit: adaptive computation, constant memory, invertible")

# memory comparison
n_layers = 50
resnet_mem = n_layers * d * d  # store all layer weights
node_mem = d * d  # single shared function
print(f"\n  ResNet ({n_layers} layers) params: {resnet_mem}")
print(f"  Neural ODE params: {node_mem} (shared across depth)")
