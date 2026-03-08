"""Rejection Sampling — generating samples from complex distributions."""

import numpy as np

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M, n_samples):
    """Sample from target using proposal * M envelope."""
    samples = []
    n_total = 0
    while len(samples) < n_samples:
        x = proposal_sample()
        u = np.random.rand()
        n_total += 1
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
    acceptance_rate = n_samples / n_total
    return np.array(samples), acceptance_rate

# --- demo ---
np.random.seed(42)

# target: mixture of two Gaussians
def target(x):
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) / (0.5 * np.sqrt(2*np.pi)) + \
           0.7 * np.exp(-0.5 * ((x + 1) / 1.0)**2) / (1.0 * np.sqrt(2*np.pi))

# proposal: wide Gaussian
def proposal_sample():
    return np.random.randn() * 3

def proposal_pdf(x):
    return np.exp(-0.5 * (x/3)**2) / (3 * np.sqrt(2*np.pi))

M = 5.0  # envelope constant
n = 5000
samples, rate = rejection_sample(target, proposal_sample, proposal_pdf, M, n)

print("=== Rejection Sampling ===")
print(f"Target: mixture of Gaussians")
print(f"Proposal: N(0, 9)")
print(f"Samples: {n}, Acceptance rate: {rate:.1%}\n")

# verify by comparing moments
print(f"Sample mean: {samples.mean():.3f}")
print(f"Sample std:  {samples.std():.3f}")

# histogram approximation
bins = np.linspace(-5, 5, 11)
counts, _ = np.histogram(samples, bins=bins, density=True)
print(f"\n{'Bin':>8} {'Empirical':>10} {'True':>10}")
print("-" * 30)
for i in range(len(counts)):
    mid = (bins[i] + bins[i+1]) / 2
    print(f"{mid:>8.1f} {counts[i]:>10.3f} {target(mid):>10.3f}")
