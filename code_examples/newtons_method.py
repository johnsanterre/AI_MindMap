"""Newton's Method — root finding and optimization."""

import numpy as np

def newtons_root(f, df, x0, tol=1e-10, max_iter=50):
    """Find root of f(x) = 0."""
    x = x0
    history = [x]
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x = x - fx / dfx
        history.append(x)
        if abs(fx) < tol:
            break
    return x, history

def newtons_optimize(f, df, ddf, x0, tol=1e-10, max_iter=50):
    """Find minimum of f(x) using Newton's method on f'(x)=0."""
    x = x0
    for i in range(max_iter):
        g = df(x)
        h = ddf(x)
        if abs(h) < 1e-15:
            break
        x = x - g / h
        if abs(g) < tol:
            break
    return x, f(x)

def newtons_multidim(f, grad, hess, x0, tol=1e-8, max_iter=50):
    """Multidimensional Newton's method for optimization."""
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        g = grad(x)
        H = hess(x)
        delta = np.linalg.solve(H, -g)
        x += delta
        if np.linalg.norm(g) < tol:
            break
    return x, f(x)

# --- demo ---
# 1) Root finding: sqrt(2) via x² - 2 = 0
root, hist = newtons_root(lambda x: x**2 - 2, lambda x: 2*x, x0=1.0)
print("=== Root Finding: x² = 2 ===")
print(f"  √2 = {root:.15f}")
print(f"  True: {np.sqrt(2):.15f}")
print(f"  Iterations: {len(hist)-1}")
print(f"  Convergence: {[f'{h:.6f}' for h in hist]}")

# 2) Optimization: minimize (x-3)² + 2
x_min, f_min = newtons_optimize(
    lambda x: (x-3)**2 + 2,
    lambda x: 2*(x-3),
    lambda x: 2,
    x0=0.0
)
print(f"\n=== 1D Optimization: (x-3)² + 2 ===")
print(f"  Minimum at x={x_min:.6f}, f(x)={f_min:.6f}")

# 3) 2D: Rosenbrock
def rosen(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_grad(x): return np.array([-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def rosen_hess(x): return np.array([[2+1200*x[0]**2-400*x[1], -400*x[0]], [-400*x[0], 200]])

x_min, f_min = newtons_multidim(rosen, rosen_grad, rosen_hess, [0.0, 0.0])
print(f"\n=== 2D Rosenbrock ===")
print(f"  Minimum at {x_min.round(6)}, f={f_min:.2e}")
