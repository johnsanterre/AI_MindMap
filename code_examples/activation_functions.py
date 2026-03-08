"""Activation Functions — implementations and properties."""

import numpy as np

def sigmoid(x):     return 1 / (1 + np.exp(-x))
def tanh(x):        return np.tanh(x)
def relu(x):        return np.maximum(0, x)
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def elu(x, a=1.0):  return np.where(x > 0, x, a * (np.exp(x) - 1))
def swish(x):       return x * sigmoid(x)
def gelu(x):        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# derivatives
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)
def relu_grad(x):    return float(x > 0)
def tanh_grad(x):    t = tanh(x); return 1 - t**2

# --- demo ---
x = np.array([-3, -1, 0, 1, 3], dtype=float)

print(f"{'x':>5}", end="")
for name in ["sigmoid", "tanh", "relu", "leaky", "elu", "swish", "gelu"]:
    print(f"{name:>9}", end="")
print()
print("-" * 70)

fns = [sigmoid, tanh, relu, leaky_relu, elu, swish, gelu]
for xi in x:
    print(f"{xi:>5.1f}", end="")
    for fn in fns:
        print(f"{fn(xi):>9.4f}", end="")
    print()

print("\nSoftmax([2.0, 1.0, 0.1]):", softmax(np.array([2.0, 1.0, 0.1])).round(3))

# vanishing gradient comparison
print("\nGradient at x=-3:")
print(f"  sigmoid: {sigmoid_grad(-3.0):.6f}")
print(f"  tanh:    {tanh_grad(-3.0):.6f}")
print(f"  relu:    {relu_grad(-3.0):.6f}")
