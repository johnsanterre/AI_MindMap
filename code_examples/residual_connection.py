"""Residual Connection — skip connections and gradient flow."""

import numpy as np

def relu(x): return np.maximum(0, x)

class PlainNet:
    def __init__(self, d, n_layers):
        self.layers = [np.random.randn(d, d) * np.sqrt(2/d) for _ in range(n_layers)]

    def forward(self, x):
        for W in self.layers:
            x = relu(x @ W)
        return x

class ResNet:
    def __init__(self, d, n_layers):
        self.layers = [np.random.randn(d, d) * np.sqrt(2/d) for _ in range(n_layers)]

    def forward(self, x):
        for W in self.layers:
            x = x + relu(x @ W)  # skip connection
        return x

def gradient_magnitude(net, x):
    """Estimate gradient magnitude via finite differences."""
    eps = 1e-5
    out = net.forward(x)
    base_norm = np.linalg.norm(out)
    grads = []
    for i in range(min(5, x.shape[1])):
        x_pert = x.copy()
        x_pert[0, i] += eps
        pert_norm = np.linalg.norm(net.forward(x_pert))
        grads.append(abs(pert_norm - base_norm) / eps)
    return np.mean(grads)

# --- demo ---
np.random.seed(42)
d = 16

print(f"{'Layers':>7} {'Plain grad':>12} {'ResNet grad':>12} {'Ratio':>8}")
print("-" * 45)
for n_layers in [2, 5, 10, 20, 50]:
    x = np.random.randn(1, d)
    plain = PlainNet(d, n_layers)
    resnet = ResNet(d, n_layers)
    g_plain = gradient_magnitude(plain, x)
    g_res = gradient_magnitude(resnet, x)
    ratio = g_res / (g_plain + 1e-15)
    print(f"{n_layers:>7} {g_plain:>12.6f} {g_res:>12.6f} {ratio:>8.1f}x")

print("\nResidual connections prevent vanishing gradients in deep networks.")
