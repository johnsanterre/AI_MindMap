"""Correlation — Pearson, Spearman, and Kendall measures of association."""

import numpy as np

def pearson(x, y):
    return np.corrcoef(x, y)[0, 1]

def spearman(x, y):
    """Rank-based correlation."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)

def kendall(x, y):
    """Kendall's tau: concordant vs discordant pairs."""
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    return (concordant - discordant) / (n * (n-1) / 2)

def partial_correlation(x, y, z):
    """Correlation between x and y controlling for z."""
    def residuals(a, b):
        slope = np.cov(a, b)[0,1] / np.var(b)
        return a - slope * b
    rx = residuals(x, z)
    ry = residuals(y, z)
    return pearson(rx, ry)

# --- demo ---
np.random.seed(42)
n = 100

print("=== Correlation Measures ===\n")

# linear relationship
x = np.random.randn(n)
y = 0.8 * x + np.random.randn(n) * 0.5
print(f"Linear: Pearson={pearson(x,y):.3f}, Spearman={spearman(x,y):.3f}, Kendall={kendall(x,y):.3f}")

# monotonic but nonlinear
x2 = np.random.uniform(0, 5, n)
y2 = np.exp(x2) + np.random.randn(n) * 2
print(f"Exp:    Pearson={pearson(x2,y2):.3f}, Spearman={spearman(x2,y2):.3f}, Kendall={kendall(x2,y2):.3f}")

# U-shaped (quadratic)
x3 = np.linspace(-3, 3, n)
y3 = x3**2 + np.random.randn(n) * 0.5
print(f"U-shape: Pearson={pearson(x3,y3):.3f}, Spearman={spearman(x3,y3):.3f}, Kendall={kendall(x3,y3):.3f}")

# spurious correlation
print("\n--- Spurious Correlation ---")
z = np.random.randn(n)  # confound
xa = z + np.random.randn(n) * 0.3
ya = z + np.random.randn(n) * 0.3
print(f"x,y corr: {pearson(xa,ya):.3f} (confounded by z)")
print(f"Partial corr (controlling z): {partial_correlation(xa, ya, z):.3f}")
