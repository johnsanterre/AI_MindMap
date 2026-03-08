"""Hypothesis Testing — t-test, chi-squared, permutation test."""

import numpy as np

def t_test(x, y):
    """Two-sample t-test (Welch's)."""
    n1, n2 = len(x), len(y)
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    t_stat = (m1 - m2) / se
    # approximate df (Welch-Satterthwaite)
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    return t_stat, df

def chi_squared_test(observed, expected):
    """Chi-squared goodness of fit."""
    chi2 = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    return chi2, df

def permutation_test(x, y, n_perms=10000):
    """Two-sample permutation test for difference in means."""
    observed_diff = abs(x.mean() - y.mean())
    combined = np.concatenate([x, y])
    count = 0
    for _ in range(n_perms):
        np.random.shuffle(combined)
        perm_diff = abs(combined[:len(x)].mean() - combined[len(x):].mean())
        if perm_diff >= observed_diff:
            count += 1
    return observed_diff, count / n_perms

# --- demo ---
np.random.seed(42)

# t-test: two groups with different means
group_a = np.random.normal(100, 15, 30)
group_b = np.random.normal(110, 15, 30)

t, df = t_test(group_a, group_b)
print("=== Welch's t-test ===")
print(f"  Group A: mean={group_a.mean():.1f}, n={len(group_a)}")
print(f"  Group B: mean={group_b.mean():.1f}, n={len(group_b)}")
print(f"  t-statistic: {t:.3f}, df: {df:.1f}")
print(f"  (|t| > 2 suggests significant difference)")

# chi-squared: is a die fair?
print("\n=== Chi-squared test ===")
rolls = np.random.choice(6, 120, p=[0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
observed = np.bincount(rolls, minlength=6)
expected = np.full(6, 20.0)
chi2, df = chi_squared_test(observed, expected)
print(f"  Observed: {observed}")
print(f"  Expected: {expected.astype(int)}")
print(f"  χ² = {chi2:.2f}, df = {df}")

# permutation test
print("\n=== Permutation test ===")
diff, p_value = permutation_test(group_a, group_b, n_perms=5000)
print(f"  Observed diff: {diff:.2f}")
print(f"  p-value: {p_value:.4f}")
