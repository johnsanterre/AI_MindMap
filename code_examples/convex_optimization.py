"""Convex Optimization — gradient descent, Newton's method, and proximal operators."""

import numpy as np

def gradient_descent(f, grad, x0, lr=0.01, n_iter=100):
    x = x0.copy()
    history = [(x.copy(), f(x))]
    for _ in range(n_iter):
        x -= lr * grad(x)
        history.append((x.copy(), f(x)))
    return x, history

def newtons_method(f, grad, hessian, x0, n_iter=20):
    x = x0.copy()
    history = [(x.copy(), f(x))]
    for _ in range(n_iter):
        H = hessian(x)
        g = grad(x)
        x -= np.linalg.solve(H, g)
        history.append((x.copy(), f(x)))
    return x, history

def proximal_gd(f, grad, prox, x0, lr=0.01, n_iter=100):
    """Proximal gradient descent for composite optimization."""
    x = x0.copy()
    history = [(x.copy(), f(x))]
    for _ in range(n_iter):
        x = prox(x - lr * grad(x), lr)
        history.append((x.copy(), f(x)))
    return x, history

# --- demo ---
np.random.seed(42)

# quadratic: f(x) = 0.5 * x^T A x - b^T x
A = np.array([[4, 1], [1, 2]])
b = np.array([1, 1])
f = lambda x: 0.5 * x @ A @ x - b @ x
grad = lambda x: A @ x - b
hess = lambda x: A

x0 = np.array([5.0, 5.0])
x_opt = np.linalg.solve(A, b)

print("=== Convex Optimization ===")
print(f"min 0.5 x^T A x - b^T x")
print(f"Optimal: x* = {x_opt.round(4)}, f* = {f(x_opt):.4f}\n")

x_gd, hist_gd = gradient_descent(f, grad, x0, lr=0.1, n_iter=50)
x_nt, hist_nt = newtons_method(f, grad, hess, x0, n_iter=10)

print(f"{'Iter':>5} {'GD f(x)':>12} {'Newton f(x)':>12}")
print("-" * 32)
for i in [0, 1, 2, 5, 10]:
    gd_val = hist_gd[i][1] if i < len(hist_gd) else hist_gd[-1][1]
    nt_val = hist_nt[i][1] if i < len(hist_nt) else hist_nt[-1][1]
    print(f"{i:>5} {gd_val:>12.6f} {nt_val:>12.6f}")

print(f"\nGD converged: {x_gd.round(4)}")
print(f"Newton converged: {x_nt.round(4)}")
print(f"Newton converges in 1 step for quadratics (exact Hessian).")
