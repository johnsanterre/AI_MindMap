"""Neural Network — 2-layer feedforward with backprop (XOR problem)."""

import numpy as np

def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)
def sigmoid(x):    return 1 / (1 + np.exp(-x))

class TwoLayerNet:
    def __init__(self, d_in, d_hidden, d_out):
        s = 0.5
        self.W1 = np.random.randn(d_in, d_hidden) * s
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_hidden, d_out) * s
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, X, y, lr=0.5):
        m = X.shape[0]
        dz2 = self.out - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * dz1.sum(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# --- demo: XOR ---
np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

net = TwoLayerNet(2, 8, 1)
for epoch in range(5000):
    out = net.forward(X)
    net.backward(X, y, lr=1.0)
    if epoch % 1000 == 0:
        loss = -np.mean(y * np.log(out+1e-8) + (1-y) * np.log(1-out+1e-8))
        print(f"Epoch {epoch:4d}  Loss={loss:.4f}")

print("\nPredictions:")
for xi, yi, pi in zip(X, y, net.forward(X)):
    print(f"  {xi} → {pi[0]:.3f} (target {yi[0]:.0f})")
