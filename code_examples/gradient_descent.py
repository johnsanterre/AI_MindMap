"""Gradient Descent — SGD, Momentum, Adam on Rosenbrock function."""

import numpy as np

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

def sgd(grad_fn, x0, lr=0.001, steps=5000):
    x = np.array(x0, dtype=float)
    for i in range(steps):
        x -= lr * grad_fn(*x)
    return x

def momentum(grad_fn, x0, lr=0.001, beta=0.9, steps=5000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    for i in range(steps):
        g = grad_fn(*x)
        v = beta * v + g
        x -= lr * v
    return x

def adam(grad_fn, x0, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, steps=5000):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = grad_fn(*x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        x -= lr * mh / (np.sqrt(vh) + eps)
    return x

# --- demo ---
x0 = [-1.0, 1.0]
print(f"Optimizing Rosenbrock from {x0} (minimum at [1,1]):\n")

result = sgd(rosenbrock_grad, x0)
print(f"SGD:      {result.round(4)}, f={rosenbrock(*result):.6f}")

result = momentum(rosenbrock_grad, x0)
print(f"Momentum: {result.round(4)}, f={rosenbrock(*result):.6f}")

result = adam(rosenbrock_grad, x0)
print(f"Adam:     {result.round(4)}, f={rosenbrock(*result):.6f}")
