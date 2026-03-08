"""AdaGrad — adaptive learning rate per parameter."""

import numpy as np

def adagrad(grad_fn, x0, lr=0.5, n_steps=50):
    x = x0.copy()
    cache = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(n_steps):
        g = grad_fn(x)
        cache += g**2
        x -= lr * g / (np.sqrt(cache) + 1e-8)
        history.append(x.copy())
    return np.array(history)

def sgd(grad_fn, x0, lr=0.1, n_steps=50):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(n_steps):
        x -= lr * grad_fn(x)
        history.append(x.copy())
    return np.array(history)

# --- demo ---
np.random.seed(42)

# elongated quadratic: f(x,y) = 50*x^2 + y^2
def grad_fn(x):
    return np.array([100 * x[0], 2 * x[1]])

x0 = np.array([1.0, 1.0])

path_sgd = sgd(grad_fn, x0, lr=0.009)
path_ada = adagrad(grad_fn, x0, lr=0.5)

print("=== AdaGrad vs SGD on elongated quadratic ===")
print(f"f(x,y) = 50x² + y²,  start = {x0}\n")
print(f"{'Step':>5} {'SGD x':>20} {'AdaGrad x':>20}")
print("-" * 48)
for i in [0, 1, 5, 10, 25, 50]:
    s = path_sgd[i]
    a = path_ada[i]
    print(f"{i:>5} ({s[0]:>7.4f}, {s[1]:>7.4f})  ({a[0]:>7.4f}, {a[1]:>7.4f})")

print("\nAdaGrad adapts LR per dimension — faster on sparse/elongated problems.")
