"""Dual Numbers Autodiff — forward-mode automatic differentiation."""

import numpy as np

class Dual:
    """Dual number: a + bε where ε² = 0."""
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + o.real, self.dual + o.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real * o.real, self.real * o.dual + self.dual * o.real)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real - o.real, self.dual - o.dual)

    def __rsub__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(o.real - self.real, o.dual - self.dual)

    def __truediv__(self, other):
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real / o.real,
                    (self.dual * o.real - self.real * o.dual) / o.real**2)

    def __pow__(self, n):
        return Dual(self.real**n, n * self.real**(n-1) * self.dual)

    def __repr__(self):
        return f"{self.real:.4f} + {self.dual:.4f}ε"

def sin(x):
    if isinstance(x, Dual):
        return Dual(np.sin(x.real), np.cos(x.real) * x.dual)
    return np.sin(x)

def cos(x):
    if isinstance(x, Dual):
        return Dual(np.cos(x.real), -np.sin(x.real) * x.dual)
    return np.cos(x)

def exp(x):
    if isinstance(x, Dual):
        e = np.exp(x.real)
        return Dual(e, e * x.dual)
    return np.exp(x)

def derivative(f, x):
    result = f(Dual(x, 1.0))
    return result.dual

# --- demo ---
print("=== Forward-Mode Autodiff with Dual Numbers ===\n")

# f(x) = x³ + 2x² - 5x + 3, f'(x) = 3x² + 4x - 5
f1 = lambda x: x**3 + 2*x**2 - 5*x + 3
print("f(x) = x³ + 2x² - 5x + 3")
for x in [-2, 0, 1, 3]:
    d = derivative(f1, x)
    exact = 3*x**2 + 4*x - 5
    print(f"  f'({x}) = {d:.4f} (exact: {exact})")

# f(x) = sin(x) * exp(-x/5)
print("\nf(x) = sin(x) * exp(-x/5)")
f2 = lambda x: sin(x) * exp(x * (-0.2))
for x in [0, 1, 2, 3]:
    d = derivative(f2, x)
    exact = np.cos(x)*np.exp(-x/5) + np.sin(x)*np.exp(-x/5)*(-0.2)
    print(f"  f'({x}) = {d:.6f} (exact: {exact:.6f})")

# verify: dual number arithmetic
print(f"\nDual(3, 1) * Dual(4, 1) = {Dual(3,1) * Dual(4,1)}")
print("  (3+ε)(4+ε) = 12 + 7ε  ← real=product, dual=sum of originals)")
