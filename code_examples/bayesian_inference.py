"""Bayesian Inference — conjugate prior and simple MCMC (Metropolis-Hastings)."""

import numpy as np

# --- Conjugate prior: Beta-Binomial ---
def beta_binomial(successes, trials, prior_a=1, prior_b=1):
    """Update Beta prior with binomial data."""
    post_a = prior_a + successes
    post_b = prior_b + (trials - successes)
    mean = post_a / (post_a + post_b)
    var = (post_a * post_b) / ((post_a + post_b)**2 * (post_a + post_b + 1))
    return post_a, post_b, mean, var

# --- Metropolis-Hastings MCMC ---
def metropolis(log_posterior, x0, proposal_std=0.5, n_samples=10000, burnin=1000):
    samples = []
    x = x0
    accepted = 0
    for i in range(n_samples + burnin):
        x_new = x + np.random.randn() * proposal_std
        log_ratio = log_posterior(x_new) - log_posterior(x)
        if np.log(np.random.random()) < log_ratio:
            x = x_new
            accepted += 1
        if i >= burnin:
            samples.append(x)
    return np.array(samples), accepted / (n_samples + burnin)

# --- demo ---
np.random.seed(42)

# 1) Beta-Binomial: coin flipping
print("=== Beta-Binomial (coin flip) ===")
print("Prior: Beta(1,1) = Uniform")
for n_flips, n_heads in [(10, 7), (100, 73), (1000, 712)]:
    a, b, mean, var = beta_binomial(n_heads, n_flips)
    ci = (mean - 1.96*np.sqrt(var), mean + 1.96*np.sqrt(var))
    print(f"  {n_flips:4d} flips, {n_heads:4d} heads → "
          f"posterior mean={mean:.3f}, 95% CI=({ci[0]:.3f}, {ci[1]:.3f})")

# 2) MCMC: estimate mean of normal distribution
print("\n=== Metropolis-Hastings MCMC ===")
data = np.random.normal(loc=3.5, scale=1.0, size=50)

def log_posterior(mu):
    # Normal likelihood + Normal(0, 10) prior on mu
    log_prior = -0.5 * (mu / 10)**2
    log_lik = -0.5 * np.sum((data - mu)**2)
    return log_prior + log_lik

samples, accept_rate = metropolis(log_posterior, x0=0.0, n_samples=10000)
print(f"True mean: 3.5")
print(f"MCMC estimate: {samples.mean():.3f} ± {samples.std():.3f}")
print(f"Acceptance rate: {accept_rate:.1%}")
