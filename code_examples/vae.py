"""Variational Autoencoder — VAE with reparameterization trick."""

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
def relu(x): return np.maximum(0, x)

class VAE:
    def __init__(self, d_in, d_hidden, d_latent):
        s = lambda n: np.sqrt(2/n)
        # encoder
        self.We1 = np.random.randn(d_in, d_hidden) * s(d_in)
        self.be1 = np.zeros(d_hidden)
        self.Wmu = np.random.randn(d_hidden, d_latent) * s(d_hidden)
        self.bmu = np.zeros(d_latent)
        self.Wlv = np.random.randn(d_hidden, d_latent) * s(d_hidden)
        self.blv = np.zeros(d_latent)
        # decoder
        self.Wd1 = np.random.randn(d_latent, d_hidden) * s(d_latent)
        self.bd1 = np.zeros(d_hidden)
        self.Wd2 = np.random.randn(d_hidden, d_in) * s(d_hidden)
        self.bd2 = np.zeros(d_in)

    def encode(self, x):
        h = relu(x @ self.We1 + self.be1)
        mu = h @ self.Wmu + self.bmu
        log_var = h @ self.Wlv + self.blv
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = np.random.randn(*mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def decode(self, z):
        h = relu(z @ self.Wd1 + self.bd1)
        return sigmoid(h @ self.Wd2 + self.bd2)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss(self, x, x_recon, mu, log_var):
        recon = -np.mean(np.sum(x * np.log(x_recon + 1e-8) + (1-x) * np.log(1-x_recon + 1e-8), axis=1))
        kl = -0.5 * np.mean(np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1))
        return recon + kl, recon, kl

# --- demo ---
np.random.seed(42)
# binary data: 4 patterns
patterns = np.array([[1,1,0,0,0,0,0,0],
                     [0,0,1,1,0,0,0,0],
                     [0,0,0,0,1,1,0,0],
                     [0,0,0,0,0,0,1,1]], dtype=float)
X = np.repeat(patterns, 25, axis=0)  # 100 samples

vae = VAE(8, 16, 2)
x_recon, mu, log_var = vae.forward(X)
total, recon, kl = vae.loss(X, x_recon, mu, log_var)
print(f"VAE Architecture: 8 → 16 → 2 (latent) → 16 → 8")
print(f"Initial loss: {total:.3f} (recon={recon:.3f}, KL={kl:.3f})")

# generate samples from prior
z_samples = np.random.randn(5, 2)
generated = vae.decode(z_samples)
print(f"\nGenerated samples from N(0,I):")
for i, g in enumerate(generated):
    print(f"  z={z_samples[i].round(2)} → {g.round(2)}")
