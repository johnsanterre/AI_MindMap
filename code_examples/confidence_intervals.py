"""Confidence Intervals — quantifying uncertainty in estimates."""

import numpy as np

def z_interval(data, confidence=0.95):
    """CI for mean with known/large-sample variance."""
    n = len(data)
    mean = data.mean()
    se = data.std(ddof=1) / np.sqrt(n)
    # z-scores for common confidence levels
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence]
    return mean - z * se, mean + z * se

def bootstrap_ci(data, stat_fn=np.mean, n_boot=5000, confidence=0.95):
    """Bootstrap confidence interval."""
    stats = np.array([stat_fn(np.random.choice(data, len(data), replace=True))
                      for _ in range(n_boot)])
    alpha = 1 - confidence
    return np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])

def proportion_ci(successes, n, confidence=0.95):
    """Wilson score interval for proportions."""
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence]
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return center - margin, center + margin

# --- demo ---
np.random.seed(42)

# sample from known distribution
true_mean = 5.0
data = np.random.normal(true_mean, 2.0, size=50)

print("=== Confidence Intervals ===")
print(f"True mean: {true_mean}, Sample mean: {data.mean():.3f}, n={len(data)}\n")

print("--- Z-interval ---")
for conf in [0.90, 0.95, 0.99]:
    lo, hi = z_interval(data, conf)
    width = hi - lo
    contains = lo <= true_mean <= hi
    print(f"  {conf:.0%} CI: ({lo:.3f}, {hi:.3f}), width={width:.3f}, contains μ: {contains}")

print("\n--- Bootstrap CI (for median) ---")
lo, hi = bootstrap_ci(data, stat_fn=np.median)
print(f"  95% CI for median: ({lo:.3f}, {hi:.3f})")
print(f"  Sample median: {np.median(data):.3f}")

print("\n--- Proportion CI ---")
# coin flip: 60 heads out of 100
lo, hi = proportion_ci(60, 100)
print(f"  60/100 success: 95% CI = ({lo:.3f}, {hi:.3f})")
lo, hi = proportion_ci(6, 10)
print(f"  6/10 success:   95% CI = ({lo:.3f}, {hi:.3f}) ← wider with less data")

# coverage simulation
print("\n--- Coverage check (1000 trials) ---")
covers = 0
for _ in range(1000):
    sample = np.random.normal(true_mean, 2.0, size=50)
    lo, hi = z_interval(sample, 0.95)
    if lo <= true_mean <= hi:
        covers += 1
print(f"  95% CI actual coverage: {covers/10:.1f}%")
