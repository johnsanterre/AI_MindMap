"""Chi-Squared Test — testing independence and goodness of fit."""

import numpy as np

def chi2_goodness_of_fit(observed, expected):
    """Test if observed frequencies match expected distribution."""
    chi2 = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    # approximate p-value (chi2 with df degrees of freedom)
    # using Wilson-Hilferty approximation
    z = (chi2/df)**(1/3) - (1 - 2/(9*df))
    z /= np.sqrt(2/(9*df))
    p = 1 - 0.5 * (1 + np.tanh(z * 0.7978 / np.sqrt(2)))
    return chi2, df, p

def chi2_independence(contingency):
    """Test independence in a contingency table."""
    observed = np.array(contingency, dtype=float)
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums * col_sums / total

    chi2 = np.sum((observed - expected)**2 / expected)
    r, c = observed.shape
    df = (r - 1) * (c - 1)

    z = (chi2/df)**(1/3) - (1 - 2/(9*df))
    z /= np.sqrt(2/(9*df))
    p = 1 - 0.5 * (1 + np.tanh(z * 0.7978 / np.sqrt(2)))
    return chi2, df, p, expected

def cramers_v(chi2, n, min_dim):
    """Effect size for chi-squared test."""
    return np.sqrt(chi2 / (n * (min_dim - 1)))

# --- demo ---
np.random.seed(42)

print("=== Chi-Squared Tests ===\n")

# goodness of fit: is a die fair?
print("--- Goodness of Fit: Fair Die? ---")
observed = np.array([18, 12, 15, 22, 16, 17])  # 100 rolls
expected = np.ones(6) * 100/6
chi2, df, p = chi2_goodness_of_fit(observed, expected)
print(f"  Observed: {observed}")
print(f"  Expected: {expected.round(1)}")
print(f"  χ²={chi2:.3f}, df={df}, p≈{p:.4f}")
print(f"  {'Not fair' if p < 0.05 else 'Consistent with fair'} (α=0.05)\n")

# independence test
print("--- Independence: Treatment vs Outcome ---")
table = np.array([
    [30, 10],   # treatment: success, failure
    [20, 20],   # control: success, failure
])
chi2, df, p, expected = chi2_independence(table)
n = table.sum()
v = cramers_v(chi2, n, min(table.shape))
print(f"  Observed:")
print(f"    Treatment: {table[0]} (success, failure)")
print(f"    Control:   {table[1]}")
print(f"  χ²={chi2:.3f}, df={df}, p≈{p:.4f}, Cramér's V={v:.3f}")
print(f"  {'Dependent' if p < 0.05 else 'Independent'} (α=0.05)")
