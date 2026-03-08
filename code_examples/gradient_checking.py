"""Gradient Checking — verify analytical gradients with numerical approximation."""

import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """Central difference gradient approximation."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def relative_error(a, b):
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)

# test functions with known gradients
def quadratic(x):
    return np.sum(x**2)
def quadratic_grad(x):
    return 2 * x

def rosenbrock(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosenbrock_grad(x):
    dx0 = -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2)
    dx1 = 200*(x[1]-x[0]**2)
    return np.array([dx0, dx1])

def softmax_cross_entropy(logits):
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return -np.log(p[0])  # loss for class 0
def softmax_ce_grad(logits):
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    g = p.copy()
    g[0] -= 1
    return g

# --- demo ---
np.random.seed(42)
print(f"{'Function':>20}  {'Max Rel Error':>15}  {'Status':>8}")
print("-" * 50)

tests = [
    ("Quadratic", quadratic, quadratic_grad, np.random.randn(5)),
    ("Rosenbrock", rosenbrock, rosenbrock_grad, np.array([0.5, 0.5])),
    ("Softmax CE", softmax_cross_entropy, softmax_ce_grad, np.random.randn(4)),
]

for name, f, grad_fn, x in tests:
    analytic = grad_fn(x)
    numeric = numerical_gradient(f, x)
    err = relative_error(analytic, numeric).max()
    status = "PASS" if err < 1e-5 else "FAIL"
    print(f"{name:>20}  {err:>15.2e}  {status:>8}")
    if name == "Quadratic":
        print(f"{'':>20}  analytic={analytic.round(4)}")
        print(f"{'':>20}  numeric ={numeric.round(4)}")
