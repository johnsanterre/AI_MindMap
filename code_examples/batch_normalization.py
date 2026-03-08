"""Batch Normalization — forward and backward pass."""

import numpy as np

class BatchNorm1D:
    def __init__(self, dim, momentum=0.9):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum

    def forward(self, x, training=True, eps=1e-5):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        self.x_norm = (x - mean) / np.sqrt(var + eps)
        return self.gamma * self.x_norm + self.beta

# --- demo ---
np.random.seed(42)

# simulate activations with different scales per feature
x = np.column_stack([
    np.random.randn(32) * 10 + 50,   # large mean, large var
    np.random.randn(32) * 0.01 - 3,  # small var, negative mean
    np.random.randn(32) * 5 + 0,     # medium var
])

bn = BatchNorm1D(3)
x_norm = bn.forward(x, training=True)

print("Before BatchNorm:")
print(f"  Mean:  {x.mean(axis=0).round(2)}")
print(f"  Std:   {x.std(axis=0).round(2)}")
print(f"\nAfter BatchNorm:")
print(f"  Mean:  {x_norm.mean(axis=0).round(4)}")
print(f"  Std:   {x_norm.std(axis=0).round(4)}")
