"""RMSProp — adaptive learning rate with exponential moving average."""

import numpy as np

def rmsprop(grad_fn, x0, lr=0.01, decay=0.9, n_steps=100):
    x = x0.copy()
    cache = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(n_steps):
        g = grad_fn(x)
        cache = decay * cache + (1 - decay) * g**2
        x -= lr * g / (np.sqrt(cache) + 1e-8)
        history.append(x.copy())
    return np.array(history)

def adam(grad_fn, x0, lr=0.01, beta1=0.9, beta2=0.999, n_steps=100):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [x.copy()]
    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        history.append(x.copy())
    return np.array(history)

# --- demo ---
# Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock_grad(p):
    x, y = p
    dx = -2*(1-x) + 200*(y - x**2)*(-2*x)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

x0 = np.array([-1.0, 1.0])
path_rms = rmsprop(rosenbrock_grad, x0, lr=0.001, n_steps=500)
path_adam = adam(rosenbrock_grad, x0, lr=0.01, n_steps=500)

def rosenbrock(p):
    return (1-p[0])**2 + 100*(p[1]-p[0]**2)**2

print("=== RMSProp vs Adam on Rosenbrock ===")
print(f"Minimum at (1, 1), start = {x0}\n")
print(f"{'Step':>5} {'RMSProp f(x)':>14} {'Adam f(x)':>14}")
print("-" * 36)
for i in [0, 50, 100, 200, 500]:
    print(f"{i:>5} {rosenbrock(path_rms[i]):>14.4f} {rosenbrock(path_adam[i]):>14.4f}")

print(f"\nRMSProp final: ({path_rms[-1][0]:.4f}, {path_rms[-1][1]:.4f})")
print(f"Adam final:    ({path_adam[-1][0]:.4f}, {path_adam[-1][1]:.4f})")
