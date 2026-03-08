"""Momentum — accelerating SGD with velocity accumulation."""

import numpy as np

def sgd(grad_fn, x0, lr=0.01, n_steps=200):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(n_steps):
        x -= lr * grad_fn(x)
        path.append(x.copy())
    return np.array(path)

def sgd_momentum(grad_fn, x0, lr=0.01, momentum=0.9, n_steps=200):
    x = x0.copy()
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_steps):
        v = momentum * v - lr * grad_fn(x)
        x += v
        path.append(x.copy())
    return np.array(path)

def nesterov_momentum(grad_fn, x0, lr=0.01, momentum=0.9, n_steps=200):
    x = x0.copy()
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_steps):
        g = grad_fn(x + momentum * v)  # lookahead
        v = momentum * v - lr * g
        x += v
        path.append(x.copy())
    return np.array(path)

# --- demo ---
np.random.seed(42)

# ill-conditioned quadratic: f(x,y) = 25x² + y²
f = lambda x: 25*x[0]**2 + x[1]**2
grad = lambda x: np.array([50*x[0], 2*x[1]])
x0 = np.array([3.0, 3.0])

path_sgd = sgd(grad, x0, lr=0.015)
path_mom = sgd_momentum(grad, x0, lr=0.015, momentum=0.9)
path_nes = nesterov_momentum(grad, x0, lr=0.015, momentum=0.9)

print("=== Momentum Methods ===")
print(f"f(x,y) = 25x² + y², start = {x0}\n")
print(f"{'Step':>5} {'SGD f(x)':>12} {'Momentum':>12} {'Nesterov':>12}")
print("-" * 44)
for i in [0, 5, 10, 25, 50, 100, 200]:
    fs = f(path_sgd[i])
    fm = f(path_mom[min(i, len(path_mom)-1)])
    fn = f(path_nes[min(i, len(path_nes)-1)])
    print(f"{i:>5} {fs:>12.4f} {fm:>12.4f} {fn:>12.4f}")

print(f"\nFinal: SGD={f(path_sgd[-1]):.6f}, Mom={f(path_mom[-1]):.6f}, Nes={f(path_nes[-1]):.6f}")
