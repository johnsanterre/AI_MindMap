"""Variational Inference — approximate Bayesian inference via optimization."""

import numpy as np

def vi_gaussian(data, n_iter=500, lr=0.01):
    """Variational inference for Gaussian with unknown mean and known variance.
    Prior: N(0, 10), Likelihood: N(mu, 1), Variational: N(m, s^2).
    """
    # variational parameters
    m = 0.0      # mean
    log_s = 0.0  # log std

    prior_mu, prior_sigma = 0.0, 10.0
    n = len(data)
    data_mean = data.mean()

    elbos = []
    for _ in range(n_iter):
        s = np.exp(log_s)
        # ELBO gradient (closed form for conjugate Gaussian)
        # E_q[log p(data|mu)] gradient
        dm_likelihood = n * (data_mean - m)
        # E_q[log p(mu)] gradient
        dm_prior = -(m - prior_mu) / prior_sigma**2
        # entropy gradient
        ds_entropy = 1.0  # d/d(log_s) of log(s)

        # KL gradient for log_s
        ds_kl = s / prior_sigma**2 - 1  # simplified

        m += lr * (dm_likelihood + dm_prior)
        log_s -= lr * (ds_kl - ds_entropy)

        # compute ELBO
        s = np.exp(log_s)
        kl = np.log(prior_sigma/s) + (s**2 + (m-prior_mu)**2)/(2*prior_sigma**2) - 0.5
        ll = -0.5 * n * (np.log(2*np.pi) + (data - m).var() + s**2)
        elbos.append(-kl + ll / n)

    return m, np.exp(log_s), elbos

# --- demo ---
np.random.seed(42)

true_mu = 3.0
data = np.random.randn(50) + true_mu

m, s, elbos = vi_gaussian(data, n_iter=500, lr=0.01)

# exact posterior (conjugate)
prior_sigma = 10.0
post_prec = 1/prior_sigma**2 + len(data)
post_var = 1 / post_prec
post_mu = post_var * (0/prior_sigma**2 + len(data) * data.mean())
post_std = np.sqrt(post_var)

print("=== Variational Inference: Gaussian Mean ===")
print(f"True μ = {true_mu}, N = {len(data)}, Data mean = {data.mean():.3f}\n")
print(f"{'':>20} {'Mean':>8} {'Std':>8}")
print("-" * 38)
print(f"{'Exact posterior':>20} {post_mu:>8.4f} {post_std:>8.4f}")
print(f"{'Variational approx':>20} {m:>8.4f} {s:>8.4f}")
print(f"\nELBO convergence: {elbos[0]:.2f} → {elbos[-1]:.2f}")
