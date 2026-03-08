"""Numerical Integration — trapezoidal, Simpson's, Gaussian quadrature."""

import numpy as np

def trapezoidal(f, a, b, n=100):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h * (y[0]/2 + y[-1]/2 + y[1:-1].sum())

def simpsons(f, a, b, n=100):
    """Simpson's 1/3 rule (n must be even)."""
    if n % 2: n += 1
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h/3 * (y[0] + y[-1] + 4*y[1::2].sum() + 2*y[2:-1:2].sum())

def gauss_legendre(f, a, b, n=5):
    """Gaussian quadrature with Legendre points."""
    # precomputed nodes and weights for small n
    nodes_weights = {
        2: ([-0.5773502692, 0.5773502692], [1.0, 1.0]),
        3: ([-0.7745966692, 0.0, 0.7745966692], [0.5555555556, 0.8888888889, 0.5555555556]),
        5: ([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459],
            [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]),
    }
    nodes, weights = nodes_weights[n]
    # transform from [-1,1] to [a,b]
    t = np.array(nodes)
    w = np.array(weights)
    x = (b-a)/2 * t + (a+b)/2
    return (b-a)/2 * np.sum(w * f(x))

# --- demo ---
print("=== Numerical Integration ===\n")

tests = [
    ("∫₀¹ x² dx", lambda x: x**2, 0, 1, 1/3),
    ("∫₀^π sin(x) dx", np.sin, 0, np.pi, 2.0),
    ("∫₀¹ e^x dx", np.exp, 0, 1, np.e - 1),
    ("∫₀¹ 4/(1+x²) dx = π", lambda x: 4/(1+x**2), 0, 1, np.pi),
]

for name, f, a, b, true in tests:
    trap = trapezoidal(f, a, b, 100)
    simp = simpsons(f, a, b, 100)
    gauss = gauss_legendre(f, a, b, 5)
    print(f"  {name}")
    print(f"    True:     {true:.10f}")
    print(f"    Trapez:   {trap:.10f}  (err={abs(trap-true):.2e})")
    print(f"    Simpson:  {simp:.10f}  (err={abs(simp-true):.2e})")
    print(f"    Gauss(5): {gauss:.10f}  (err={abs(gauss-true):.2e})")
    print()
