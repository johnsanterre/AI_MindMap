"""ANOVA — testing differences across multiple groups."""

import numpy as np

def one_way_anova(*groups):
    """One-way ANOVA: are group means different?"""
    k = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)
    grand_mean = np.concatenate(groups).mean()

    # between-group sum of squares
    ss_between = sum(n * (g.mean() - grand_mean)**2 for g, n in zip(groups, ns))
    df_between = k - 1

    # within-group sum of squares
    ss_within = sum(np.sum((g - g.mean())**2) for g in groups)
    df_within = N - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # eta-squared (effect size)
    eta2 = ss_between / (ss_between + ss_within)

    return f_stat, df_between, df_within, eta2

def tukey_hsd(groups, alpha=0.05):
    """Pairwise comparisons with pooled std error."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    ms_within = sum(np.sum((g - g.mean())**2) for g in groups) / (N - k)

    comparisons = []
    for i in range(k):
        for j in range(i+1, k):
            diff = abs(groups[i].mean() - groups[j].mean())
            se = np.sqrt(ms_within * (1/len(groups[i]) + 1/len(groups[j])) / 2)
            q = diff / se
            comparisons.append((i, j, diff, q))
    return comparisons

# --- demo ---
np.random.seed(42)

# three teaching methods, test scores
method_a = np.random.normal(75, 8, 30)
method_b = np.random.normal(80, 8, 30)
method_c = np.random.normal(78, 8, 30)

f, df1, df2, eta2 = one_way_anova(method_a, method_b, method_c)

print("=== One-Way ANOVA ===")
print(f"Comparing 3 teaching methods (n=30 each)\n")
print(f"  Method A: mean={method_a.mean():.1f}, std={method_a.std():.1f}")
print(f"  Method B: mean={method_b.mean():.1f}, std={method_b.std():.1f}")
print(f"  Method C: mean={method_c.mean():.1f}, std={method_c.std():.1f}")
print(f"\n  F({df1},{df2}) = {f:.3f}")
print(f"  η² = {eta2:.3f} (effect size)")
print(f"  {'Significant' if f > 3.1 else 'Not significant'} at α=0.05 (F_crit ≈ 3.1)")

print(f"\n--- Post-hoc Pairwise Comparisons ---")
comps = tukey_hsd([method_a, method_b, method_c])
labels = ["A", "B", "C"]
for i, j, diff, q in comps:
    print(f"  {labels[i]} vs {labels[j]}: diff={diff:.2f}, q={q:.3f}")
