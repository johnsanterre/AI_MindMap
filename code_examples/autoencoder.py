"""Autoencoder — simple numpy autoencoder for dimensionality reduction."""

import numpy as np

def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)

class Autoencoder:
    def __init__(self, d_input, d_hidden):
        s = np.sqrt(2.0 / d_input)
        self.W_enc = np.random.randn(d_input, d_hidden) * s
        self.b_enc = np.zeros(d_hidden)
        self.W_dec = np.random.randn(d_hidden, d_input) * s
        self.b_dec = np.zeros(d_input)

    def encode(self, X):
        self._z = X @ self.W_enc + self.b_enc
        self._h = relu(self._z)
        return self._h

    def decode(self, H):
        return H @ self.W_dec + self.b_dec

    def forward(self, X):
        self._X = X
        H = self.encode(X)
        return self.decode(H)

    def train_step(self, X, lr=0.001):
        m = len(X)
        X_hat = self.forward(X)
        loss = np.mean((X - X_hat) ** 2)

        # backward
        dX_hat = (2/m) * (X_hat - X)
        self.W_dec -= lr * (self._h.T @ dX_hat)
        self.b_dec -= lr * dX_hat.sum(axis=0)
        dh = dX_hat @ self.W_dec.T * relu_grad(self._z)
        self.W_enc -= lr * (self._X.T @ dh)
        self.b_enc -= lr * dh.sum(axis=0)
        return loss

# --- demo ---
np.random.seed(42)
# 10D data living on a 3D manifold
t1, t2, t3 = np.random.randn(3, 200)
X = np.column_stack([t1, t2, t3, t1+t2, t1-t3, 2*t2,
                     t3-t1, t1+t2+t3, 2*t1-t2, t2+t3])
X += np.random.randn(*X.shape) * 0.1

ae = Autoencoder(10, 3)
for epoch in range(2001):
    loss = ae.train_step(X, lr=0.0005)
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d}  MSE={loss:.4f}")

encoded = ae.encode(X)
reconstructed = ae.decode(encoded)
final_mse = np.mean((X - reconstructed) ** 2)
print(f"\n10D → 3D → 10D reconstruction MSE: {final_mse:.4f}")
print(f"Encoded shape: {encoded.shape}")
