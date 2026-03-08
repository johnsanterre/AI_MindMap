"""Monte Carlo Methods — estimation, integration, and simulation."""

import numpy as np

def estimate_pi(n=100000):
    """Estimate π by throwing darts at a unit square."""
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = (x**2 + y**2) <= 1
    return 4 * inside.mean()

def mc_integrate(f, a, b, n=100000):
    """Monte Carlo integration of f over [a, b]."""
    x = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(x))

def importance_sampling(f, p_sample, p_pdf, q_pdf, n=100000):
    """Estimate E_p[f(x)] using samples from q."""
    x = p_sample(n)
    weights = p_pdf(x) / q_pdf(x)
    return np.mean(f(x) * weights)

def bootstrap_mean_ci(data, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval for the mean."""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int(alpha/2 * n_bootstrap)]
    hi = means[int((1-alpha/2) * n_bootstrap)]
    return np.mean(means), lo, hi

# --- demo ---
np.random.seed(42)

# 1) Estimate pi
pi_est = estimate_pi(1000000)
print(f"π estimate (1M samples): {pi_est:.5f} (true: {np.pi:.5f})")

# 2) Monte Carlo integration: ∫₀¹ sin(πx) dx = 2/π
integral = mc_integrate(lambda x: np.sin(np.pi * x), 0, 1, 500000)
print(f"∫₀¹ sin(πx)dx = {integral:.5f} (true: {2/np.pi:.5f})")

# 3) Bootstrap CI
data = np.random.exponential(scale=2.0, size=50)
mean, lo, hi = bootstrap_mean_ci(data)
print(f"\nBootstrap mean: {mean:.3f}, 95% CI: ({lo:.3f}, {hi:.3f})")
print(f"Sample mean: {data.mean():.3f}, True mean: 2.0")
