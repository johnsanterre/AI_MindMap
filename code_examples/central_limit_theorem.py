"""Central Limit Theorem — sample means converge to normal regardless of distribution."""

import numpy as np

def sample_means(distribution_fn, n_samples, sample_size, n_experiments):
    """Draw n_experiments sets of sample_size samples, return their means."""
    means = np.array([distribution_fn(sample_size).mean() for _ in range(n_experiments)])
    return means

def normality_test(data):
    """Simple test: compare to normal quantiles (Shapiro-like)."""
    sorted_data = np.sort(data)
    n = len(data)
    # just check skewness and kurtosis
    mean, std = data.mean(), data.std()
    z = (data - mean) / std
    skewness = np.mean(z**3)
    kurtosis = np.mean(z**4) - 3
    return skewness, kurtosis

# --- demo ---
np.random.seed(42)

distributions = {
    "Uniform(0,1)": lambda n: np.random.uniform(0, 1, n),
    "Exponential(1)": lambda n: np.random.exponential(1, n),
    "Bernoulli(0.3)": lambda n: (np.random.rand(n) < 0.3).astype(float),
    "Poisson(3)": lambda n: np.random.poisson(3, n).astype(float),
}

print("=== Central Limit Theorem ===\n")
print(f"{'Distribution':>16} {'n':>4} {'Mean of means':>14} {'Std of means':>13} {'Skewness':>9} {'Kurtosis':>9}")
print("-" * 70)

for name, dist_fn in distributions.items():
    for n in [1, 5, 30, 100]:
        means = sample_means(dist_fn, 1000, n, 1000)
        skew, kurt = normality_test(means)
        print(f"{name:>16} {n:>4} {means.mean():>14.4f} {means.std():>13.4f} {skew:>9.3f} {kurt:>9.3f}")
    print()

print("As sample size n grows:")
print("  - Skewness → 0 (symmetric)")
print("  - Kurtosis → 0 (normal-like tails)")
print("  - Std of means ≈ σ/√n")
