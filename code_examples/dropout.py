"""Dropout — regularization via random neuron deactivation."""

import numpy as np

def relu(x): return np.maximum(0, x)

class DropoutNet:
    def __init__(self, dims, drop_rate=0.5):
        self.weights = []
        self.biases = []
        self.drop_rate = drop_rate
        for i in range(len(dims)-1):
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i]))
            self.biases.append(np.zeros(dims[i+1]))

    def forward(self, X, training=True):
        self.masks = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            a = relu(z) if i < len(self.weights)-1 else z
            if training and i < len(self.weights)-1:
                mask = (np.random.rand(*a.shape) > self.drop_rate) / (1 - self.drop_rate)
                a *= mask
                self.masks.append(mask)
        return a

# --- demo ---
np.random.seed(42)
net = DropoutNet([10, 64, 32, 1], drop_rate=0.5)
X = np.random.randn(5, 10)

print("Same input, different dropout masks (training mode):")
for i in range(3):
    out = net.forward(X, training=True)
    active = [int((m > 0).sum()) for m in net.masks]
    print(f"  Run {i+1}: output={out[:3, 0].round(2)}, active neurons={active}")

print("\nInference mode (no dropout, consistent output):")
for i in range(3):
    out = net.forward(X, training=False)
    print(f"  Run {i+1}: output={out[:3, 0].round(2)}")
