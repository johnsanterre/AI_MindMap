"""Expectation-Maximization — fitting a Gaussian Mixture Model."""

import numpy as np

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def em_gmm_1d(data, k=2, max_iter=100, tol=1e-6):
    n = len(data)
    # initialize
    mu = np.random.choice(data, k, replace=False)
    sigma = np.full(k, data.std())
    pi = np.full(k, 1.0 / k)

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        resp = np.zeros((n, k))
        for j in range(k):
            resp[:, j] = pi[j] * gaussian_pdf(data, mu[j], sigma[j])
        resp /= resp.sum(axis=1, keepdims=True)

        # M-step: update parameters
        Nk = resp.sum(axis=0)
        mu_new = (resp * data[:, None]).sum(axis=0) / Nk
        sigma_new = np.sqrt((resp * (data[:, None] - mu_new)**2).sum(axis=0) / Nk)
        pi_new = Nk / n

        # check convergence
        if np.abs(mu_new - mu).max() < tol:
            print(f"  Converged at iteration {iteration}")
            break
        mu, sigma, pi = mu_new, sigma_new, pi_new

    return mu, sigma, pi, resp

# --- demo ---
np.random.seed(42)
# generate data from 3 Gaussians
data = np.concatenate([
    np.random.normal(-3, 0.5, 100),
    np.random.normal(0, 1.0, 150),
    np.random.normal(4, 0.8, 100),
])
np.random.shuffle(data)

print("EM for Gaussian Mixture Model (k=3):")
mu, sigma, pi, resp = em_gmm_1d(data, k=3)

print(f"\nLearned parameters:")
for j in range(3):
    print(f"  Component {j+1}: μ={mu[j]:+.2f}, σ={sigma[j]:.2f}, π={pi[j]:.3f}")

labels = resp.argmax(axis=1)
print(f"\nCluster sizes: {[int((labels==j).sum()) for j in range(3)]}")
