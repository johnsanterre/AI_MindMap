"""Diffusion Model — simplified denoising diffusion (1D Gaussian)."""

import numpy as np

def noise_schedule(T, beta_start=0.0001, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar

def forward_diffusion(x0, t, alpha_bar):
    """Add noise to data: q(x_t | x_0)."""
    noise = np.random.randn(*x0.shape)
    x_t = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * noise
    return x_t, noise

class SimpleDenoiser:
    """Linear denoiser: predict noise from noisy input + timestep."""
    def __init__(self, dim, T):
        self.W = np.random.randn(dim + 1, dim) * 0.01  # +1 for timestep
        self.T = T

    def predict(self, x_t, t):
        t_feat = np.full((len(x_t), 1), t / self.T)
        inp = np.hstack([x_t, t_feat])
        return inp @ self.W

    def train_step(self, x0, alpha_bar, lr=0.001):
        t = np.random.randint(1, len(alpha_bar))
        x_t, noise = forward_diffusion(x0, t, alpha_bar)
        pred = self.predict(x_t, t)
        loss = np.mean((pred - noise)**2)
        # gradient
        t_feat = np.full((len(x_t), 1), t / self.T)
        inp = np.hstack([x_t, t_feat])
        grad = 2 * inp.T @ (pred - noise) / len(x0)
        self.W -= lr * grad
        return loss

def sample(denoiser, betas, alphas, alpha_bar, n_samples=100):
    """Reverse diffusion sampling."""
    T = len(betas)
    x = np.random.randn(n_samples, 1)
    for t in range(T-1, 0, -1):
        pred_noise = denoiser.predict(x, t)
        x = (1/np.sqrt(alphas[t])) * (x - betas[t]/np.sqrt(1-alpha_bar[t]) * pred_noise)
        if t > 1:
            x += np.sqrt(betas[t]) * np.random.randn(*x.shape)
    return x

# --- demo ---
np.random.seed(42)
T = 100
betas, alphas, alpha_bar = noise_schedule(T)

# target: mixture of two Gaussians
x0 = np.concatenate([np.random.randn(200, 1) - 3, np.random.randn(200, 1) + 3])

denoiser = SimpleDenoiser(1, T)
for epoch in range(1001):
    loss = denoiser.train_step(x0, alpha_bar, lr=0.01)
    if epoch % 200 == 0:
        print(f"  Epoch {epoch:4d}  Loss={loss:.4f}")

samples = sample(denoiser, betas, alphas, alpha_bar, 1000)
print(f"\nTarget: bimodal at -3 and +3")
print(f"Samples: mean={samples.mean():.2f}, std={samples.std():.2f}")
print(f"  Left mode (< 0):  mean={samples[samples < 0].mean():.2f}, count={(samples < 0).sum()}")
print(f"  Right mode (> 0): mean={samples[samples > 0].mean():.2f}, count={(samples > 0).sum()}")
