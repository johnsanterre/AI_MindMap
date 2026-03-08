"""t-Test — comparing means with small samples and unknown variance."""

import numpy as np

def one_sample_t(data, mu0=0):
    n = len(data)
    t_stat = (data.mean() - mu0) / (data.std(ddof=1) / np.sqrt(n))
    df = n - 1
    # approximate p-value using large-df normal approx
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) * 0.7978 / np.sqrt(2))))
    return t_stat, df, p_value

def two_sample_t(data1, data2, equal_var=True):
    n1, n2 = len(data1), len(data2)
    m1, m2 = data1.mean(), data2.mean()
    s1, s2 = data1.std(ddof=1), data2.std(ddof=1)

    if equal_var:
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        t_stat = (m1 - m2) / (sp * np.sqrt(1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:  # Welch's t-test
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        t_stat = (m1 - m2) / se
        df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

    p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) * 0.7978 / np.sqrt(2))))
    return t_stat, df, p_value

def paired_t(data1, data2):
    return one_sample_t(data1 - data2, mu0=0)

def effect_size(data1, data2):
    """Cohen's d."""
    pooled_std = np.sqrt((data1.std(ddof=1)**2 + data2.std(ddof=1)**2) / 2)
    return (data1.mean() - data2.mean()) / pooled_std

# --- demo ---
np.random.seed(42)

print("=== t-Tests ===\n")

# one-sample: is mean different from 0?
data = np.random.normal(0.5, 1.0, 25)
t, df, p = one_sample_t(data, mu0=0)
print(f"One-sample t-test (H₀: μ=0)")
print(f"  n={len(data)}, mean={data.mean():.3f}, t={t:.3f}, df={df}, p≈{p:.4f}")
print(f"  {'Reject' if p < 0.05 else 'Fail to reject'} H₀ at α=0.05\n")

# two-sample: are two groups different?
group_a = np.random.normal(10, 2, 30)
group_b = np.random.normal(11, 2, 30)
t, df, p = two_sample_t(group_a, group_b)
d = effect_size(group_a, group_b)
print(f"Two-sample t-test")
print(f"  A: mean={group_a.mean():.2f}, B: mean={group_b.mean():.2f}")
print(f"  t={t:.3f}, df={df}, p≈{p:.4f}, Cohen's d={d:.3f}")
print(f"  {'Reject' if p < 0.05 else 'Fail to reject'} H₀ at α=0.05\n")

# paired t-test: before/after
before = np.random.normal(100, 10, 20)
after = before + np.random.normal(3, 5, 20)  # treatment effect ~3
t, df, p = paired_t(before, after)
print(f"Paired t-test (before/after)")
print(f"  Mean diff: {(after-before).mean():.2f}")
print(f"  t={t:.3f}, df={df}, p≈{p:.4f}")
