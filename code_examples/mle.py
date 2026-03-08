"""Maximum Likelihood Estimation — analytic and numerical MLE."""

import numpy as np

def mle_normal(data):
    """Analytic MLE for Normal distribution."""
    mu = data.mean()
    sigma = data.std()  # MLE uses n, not n-1
    return mu, sigma

def mle_exponential(data):
    """Analytic MLE for Exponential distribution."""
    return data.mean()  # lambda_hat = 1/mean

def mle_bernoulli(data):
    """Analytic MLE for Bernoulli."""
    return data.mean()

def numerical_mle_normal(data, lr=0.01, epochs=1000):
    """Gradient ascent on log-likelihood."""
    mu, sigma = 0.0, 1.0
    n = len(data)
    for _ in range(epochs):
        # gradients of log-likelihood
        dmu = np.sum(data - mu) / sigma**2
        dsigma = -n/sigma + np.sum((data - mu)**2) / sigma**3
        mu += lr * dmu
        sigma += lr * dsigma
        sigma = max(sigma, 0.01)
    return mu, sigma

def log_likelihood_normal(data, mu, sigma):
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - np.sum((data-mu)**2) / (2*sigma**2)

# --- demo ---
np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=200)

mu_a, sig_a = mle_normal(data)
mu_n, sig_n = numerical_mle_normal(data, lr=0.001, epochs=2000)

print("=== Normal Distribution MLE ===")
print(f"  True:      μ=5.0, σ=2.0")
print(f"  Analytic:  μ={mu_a:.3f}, σ={sig_a:.3f}")
print(f"  Numerical: μ={mu_n:.3f}, σ={sig_n:.3f}")
print(f"  Log-lik:   {log_likelihood_normal(data, mu_a, sig_a):.2f}")

print("\n=== Exponential MLE ===")
exp_data = np.random.exponential(scale=3.0, size=200)
lam = mle_exponential(exp_data)
print(f"  True: λ=3.0, MLE: λ={lam:.3f}")

print("\n=== Bernoulli MLE ===")
coin = np.random.binomial(1, 0.7, size=200)
print(f"  True: p=0.7, MLE: p={mle_bernoulli(coin):.3f}")
