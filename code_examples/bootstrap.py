"""Bootstrap — confidence intervals for any statistic."""

import numpy as np

def bootstrap_ci(data, stat_fn, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval for an arbitrary statistic."""
    stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        stats.append(stat_fn(sample))
    stats = np.sort(stats)
    lo = stats[int(alpha/2 * n_bootstrap)]
    hi = stats[int((1-alpha/2) * n_bootstrap)]
    return np.mean(stats), lo, hi

def bootstrap_paired_test(x, y, n_bootstrap=10000):
    """Bootstrap test for paired difference."""
    diffs = x - y
    observed = diffs.mean()
    count = 0
    centered = diffs - observed
    for _ in range(n_bootstrap):
        sample = np.random.choice(centered, size=len(centered), replace=True)
        if abs(sample.mean()) >= abs(observed):
            count += 1
    return observed, count / n_bootstrap

# --- demo ---
np.random.seed(42)

# skewed data (bootstrap shines here vs normal CI)
data = np.random.exponential(scale=2.0, size=50)

print("=== Bootstrap Confidence Intervals ===")
print(f"  Sample size: {len(data)}")

for name, fn in [("Mean", np.mean), ("Median", np.median),
                 ("Std Dev", np.std), ("25th %ile", lambda x: np.percentile(x, 25))]:
    est, lo, hi = bootstrap_ci(data, fn)
    print(f"  {name:>10}: {est:.3f}  95% CI: ({lo:.3f}, {hi:.3f})")

# paired test
print("\n=== Bootstrap Paired Test ===")
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 10, 30)  # treatment effect ~5
diff, p = bootstrap_paired_test(after, before)
print(f"  Mean difference: {diff:.2f}")
print(f"  p-value: {p:.4f}")
print(f"  Significant at 0.05? {'Yes' if p < 0.05 else 'No'}")
