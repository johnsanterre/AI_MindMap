"""GAN — Generative Adversarial Network (1D Gaussian)."""

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
def relu(x): return np.maximum(0, x)

class Generator:
    def __init__(self, z_dim=1, hidden=16, out=1):
        self.W1 = np.random.randn(z_dim, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, out) * 0.5
        self.b2 = np.zeros(out)

    def forward(self, z):
        self.z = z
        self.h = relu(z @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

class Discriminator:
    def __init__(self, in_dim=1, hidden=16):
        self.W1 = np.random.randn(in_dim, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * 0.5
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.x = x
        self.h = relu(x @ self.W1 + self.b1)
        return sigmoid(self.h @ self.W2 + self.b2)

# --- demo ---
np.random.seed(42)
real_mean, real_std = 5.0, 1.0
G = Generator(z_dim=1, hidden=32, out=1)
D = Discriminator(in_dim=1, hidden=32)
lr_g, lr_d = 0.01, 0.01
batch = 64

for step in range(2001):
    # real data
    real = np.random.randn(batch, 1) * real_std + real_mean
    # fake data
    z = np.random.randn(batch, 1)
    fake = G.forward(z)

    # discriminator scores
    d_real = D.forward(real)
    d_fake = D.forward(fake)

    # update D: maximize log(D(real)) + log(1 - D(fake))
    d_loss_real = -np.log(d_real + 1e-8).mean()
    d_loss_fake = -np.log(1 - d_fake + 1e-8).mean()

    # simplified gradient updates
    D.W2 += lr_d * (D.h.T @ (1 - d_real) - D.h.T @ d_fake) / batch
    D.b2 += lr_d * ((1 - d_real).mean() - d_fake.mean())

    # update G: maximize log(D(fake))
    d_fake2 = D.forward(G.forward(z))
    G.W2 += lr_g * (G.h.T @ (d_fake2 * (1 - d_fake2) * D.W2.T[:, :1])) / batch

    if step % 500 == 0:
        samples = G.forward(np.random.randn(1000, 1))
        print(f"Step {step:4d}  D_real={d_real.mean():.3f}  D_fake={d_fake.mean():.3f}  "
              f"G_mean={samples.mean():.2f}  G_std={samples.std():.2f}")

print(f"\nTarget: mean={real_mean}, std={real_std}")
