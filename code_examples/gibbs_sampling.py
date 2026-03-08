"""Gibbs Sampling — MCMC by sampling each variable conditioned on the rest."""

import numpy as np

def gibbs_bivariate_normal(mu, cov, n_samples=5000, burnin=500):
    """Gibbs sampler for bivariate normal distribution."""
    rho = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    s1, s2 = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

    samples = np.zeros((n_samples + burnin, 2))
    x, y = 0.0, 0.0

    for t in range(n_samples + burnin):
        # sample x | y
        cond_mu_x = mu[0] + rho * s1/s2 * (y - mu[1])
        cond_std_x = s1 * np.sqrt(1 - rho**2)
        x = np.random.randn() * cond_std_x + cond_mu_x

        # sample y | x
        cond_mu_y = mu[1] + rho * s2/s1 * (x - mu[0])
        cond_std_y = s2 * np.sqrt(1 - rho**2)
        y = np.random.randn() * cond_std_y + cond_mu_y

        samples[t] = [x, y]

    return samples[burnin:]

# --- demo ---
np.random.seed(42)

mu = np.array([1.0, 2.0])
cov = np.array([[1.0, 0.8], [0.8, 1.5]])

samples = gibbs_bivariate_normal(mu, cov, n_samples=10000, burnin=1000)

print("=== Gibbs Sampling: Bivariate Normal ===")
print(f"True mean: {mu}")
print(f"True cov:\n  {cov[0]}\n  {cov[1]}\n")
print(f"Sample mean: [{samples[:,0].mean():.3f}, {samples[:,1].mean():.3f}]")
sample_cov = np.cov(samples.T)
print(f"Sample cov:")
print(f"  [{sample_cov[0,0]:.3f}, {sample_cov[0,1]:.3f}]")
print(f"  [{sample_cov[1,0]:.3f}, {sample_cov[1,1]:.3f}]")

# check correlation
rho_true = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
rho_sample = np.corrcoef(samples.T)[0,1]
print(f"\nTrue correlation: {rho_true:.3f}")
print(f"Sample correlation: {rho_sample:.3f}")
