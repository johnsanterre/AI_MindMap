"""Weight Initialization — Xavier, He, and their effects on training."""

import numpy as np

def relu(x): return np.maximum(0, x)

def forward_pass(X, weights, activation=relu):
    a = X
    activations = [a]
    for W in weights:
        a = activation(a @ W)
        activations.append(a)
    return activations

def init_zeros(layers):
    return [np.zeros((layers[i], layers[i+1])) for i in range(len(layers)-1)]

def init_random(layers, scale=1.0):
    return [np.random.randn(layers[i], layers[i+1]) * scale for i in range(len(layers)-1)]

def init_xavier(layers):
    """Xavier/Glorot: good for tanh/sigmoid."""
    return [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / (layers[i] + layers[i+1]))
            for i in range(len(layers)-1)]

def init_he(layers):
    """He/Kaiming: good for ReLU."""
    return [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            for i in range(len(layers)-1)]

# --- demo ---
np.random.seed(42)
layers = [256, 256, 256, 256, 256, 256]  # 5 hidden layers
X = np.random.randn(100, 256)

print(f"Activation stats through a 5-layer ReLU network:\n")
print(f"{'Init':>12} {'Layer 1':>10} {'Layer 3':>10} {'Layer 5':>10} {'Dead%':>8}")
print("-" * 55)

for name, init_fn in [("Zeros", init_zeros), ("N(0,1)", lambda l: init_random(l, 1.0)),
                       ("N(0,0.01)", lambda l: init_random(l, 0.01)),
                       ("Xavier", init_xavier), ("He", init_he)]:
    W = init_fn(layers)
    acts = forward_pass(X, W)
    stds = [a.std() for a in acts[1:]]
    dead = (acts[-1] == 0).mean() * 100
    print(f"{name:>12} {stds[0]:>10.4f} {stds[2]:>10.4f} {stds[4]:>10.4f} {dead:>7.1f}%")
