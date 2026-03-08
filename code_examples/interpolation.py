"""Interpolation — Lagrange, Newton, and spline interpolation."""

import numpy as np

def lagrange(x_pts, y_pts, x):
    """Lagrange polynomial interpolation."""
    n = len(x_pts)
    result = 0.0
    for i in range(n):
        term = y_pts[i]
        for j in range(n):
            if i != j:
                term *= (x - x_pts[j]) / (x_pts[i] - x_pts[j])
        result += term
    return result

def newton_divided_diff(x_pts, y_pts):
    """Compute Newton's divided difference coefficients."""
    n = len(x_pts)
    coef = y_pts.copy().astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_pts[i] - x_pts[i-j])
    return coef

def newton_eval(coef, x_pts, x):
    """Evaluate Newton polynomial at x."""
    n = len(coef)
    result = coef[-1]
    for i in range(n-2, -1, -1):
        result = result * (x - x_pts[i]) + coef[i]
    return result

def linear_spline(x_pts, y_pts, x):
    """Piecewise linear interpolation."""
    i = np.searchsorted(x_pts, x) - 1
    i = np.clip(i, 0, len(x_pts)-2)
    t = (x - x_pts[i]) / (x_pts[i+1] - x_pts[i])
    return y_pts[i] + t * (y_pts[i+1] - y_pts[i])

# --- demo ---
# interpolate sin(x) from sample points
x_pts = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
y_pts = np.sin(x_pts)

test_x = np.array([np.pi/6, np.pi/3, 2*np.pi/3, 5*np.pi/6])
true_y = np.sin(test_x)

coef = newton_divided_diff(x_pts, y_pts)

print("Interpolating sin(x) from 5 sample points:\n")
print(f"{'x':>8} {'True':>10} {'Lagrange':>10} {'Newton':>10} {'Linear':>10}")
print("-" * 52)
for x, y_true in zip(test_x, true_y):
    y_lag = lagrange(x_pts, y_pts, x)
    y_new = newton_eval(coef, x_pts, x)
    y_lin = linear_spline(x_pts, y_pts, x)
    print(f"{x:>8.4f} {y_true:>10.6f} {y_lag:>10.6f} {y_new:>10.6f} {y_lin:>10.6f}")

print(f"\nMax error (Lagrange): {max(abs(lagrange(x_pts, y_pts, x) - np.sin(x)) for x in test_x):.6f}")
