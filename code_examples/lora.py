"""LoRA — Low-Rank Adaptation for efficient fine-tuning."""

import numpy as np

class LoRALayer:
    """Simulates W + B@A low-rank update to a frozen weight matrix."""
    def __init__(self, d_in, d_out, rank=4, alpha=1.0):
        self.W_frozen = np.random.randn(d_in, d_out) * 0.1  # pretrained (frozen)
        self.A = np.random.randn(d_in, rank) * 0.01         # trainable
        self.B = np.zeros((rank, d_out))                      # trainable (init zero)
        self.scale = alpha / rank

    def forward(self, x):
        return x @ self.W_frozen + self.scale * (x @ self.A @ self.B)

    def trainable_params(self):
        return self.A.size + self.B.size

    def total_params(self):
        return self.W_frozen.size + self.trainable_params()

    def train_step(self, x, target, lr=0.01):
        """Simple gradient descent on MSE."""
        pred = self.forward(x)
        loss = np.mean((pred - target)**2)
        # gradient w.r.t. B and A
        err = 2 * (pred - target) / target.size
        dB = (self.scale * (x @ self.A)).T @ err
        dA = self.scale * x.T @ (err @ self.B.T)
        self.B -= lr * dB
        self.A -= lr * dA
        return loss

# --- demo ---
np.random.seed(42)
d_in, d_out, rank = 64, 32, 4

layer = LoRALayer(d_in, d_out, rank=rank)
print(f"Total params:     {layer.total_params():,}")
print(f"Trainable (LoRA): {layer.trainable_params():,} ({layer.trainable_params()/layer.total_params():.1%})")
print(f"Frozen:           {layer.W_frozen.size:,}")
print(f"Rank: {rank}\n")

# simulate fine-tuning
X = np.random.randn(100, d_in)
# target = original output + small task-specific shift
target = X @ layer.W_frozen + X @ (np.random.randn(d_in, d_out) * 0.02)

for epoch in range(201):
    loss = layer.train_step(X, target, lr=0.01)
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}  Loss={loss:.6f}")
