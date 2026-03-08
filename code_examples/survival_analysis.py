"""Survival Analysis — Kaplan-Meier estimator and hazard rates."""

import numpy as np

def kaplan_meier(times, events):
    """Kaplan-Meier survival curve estimate.
    times: observed times, events: 1=event occurred, 0=censored."""
    unique_times = np.sort(np.unique(times[events == 1]))
    n_at_risk = len(times)
    survival = 1.0
    curve = [(0, 1.0)]

    for t in unique_times:
        d = np.sum((times == t) & (events == 1))  # events at t
        c = np.sum((times == t) & (events == 0))  # censored at t
        survival *= (1 - d / n_at_risk)
        curve.append((t, survival))
        n_at_risk -= (d + c)

    return np.array(curve)

def nelson_aalen(times, events):
    """Nelson-Aalen cumulative hazard estimator."""
    unique_times = np.sort(np.unique(times[events == 1]))
    n_at_risk = len(times)
    cum_hazard = 0.0
    curve = [(0, 0.0)]

    for t in unique_times:
        d = np.sum((times == t) & (events == 1))
        c = np.sum((times == t) & (events == 0))
        cum_hazard += d / n_at_risk
        curve.append((t, cum_hazard))
        n_at_risk -= (d + c)

    return np.array(curve)

def log_rank_test(times1, events1, times2, events2):
    """Compare two survival curves."""
    all_times = np.sort(np.unique(np.concatenate([
        times1[events1 == 1], times2[events2 == 1]
    ])))
    O1, E1 = 0, 0
    for t in all_times:
        n1 = np.sum(times1 >= t)
        n2 = np.sum(times2 >= t)
        d1 = np.sum((times1 == t) & (events1 == 1))
        d2 = np.sum((times2 == t) & (events2 == 1))
        d = d1 + d2
        n = n1 + n2
        if n > 0:
            O1 += d1
            E1 += n1 * d / n
    chi2 = (O1 - E1)**2 / (E1 + 1e-8)
    return chi2

# --- demo ---
np.random.seed(42)

# group A: baseline survival
times_a = np.random.exponential(10, 50)
events_a = (np.random.rand(50) > 0.2).astype(int)

# group B: better survival
times_b = np.random.exponential(15, 50)
events_b = (np.random.rand(50) > 0.2).astype(int)

km_a = kaplan_meier(times_a, events_a)
km_b = kaplan_meier(times_b, events_b)

print("=== Survival Analysis ===\n")
print("--- Kaplan-Meier Curves ---")
print(f"{'Time':>6} {'S(t) Group A':>13} {'S(t) Group B':>13}")
print("-" * 34)
for t in [0, 2, 5, 8, 10, 15, 20]:
    sa = km_a[km_a[:, 0] <= t][-1, 1] if len(km_a[km_a[:, 0] <= t]) > 0 else 1.0
    sb = km_b[km_b[:, 0] <= t][-1, 1] if len(km_b[km_b[:, 0] <= t]) > 0 else 1.0
    print(f"{t:>6} {sa:>13.3f} {sb:>13.3f}")

# median survival
for name, km in [("A", km_a), ("B", km_b)]:
    median_idx = np.where(km[:, 1] <= 0.5)[0]
    median = km[median_idx[0], 0] if len(median_idx) > 0 else ">max"
    print(f"\nMedian survival Group {name}: {median}")

chi2 = log_rank_test(times_a, events_a, times_b, events_b)
print(f"\nLog-rank test χ² = {chi2:.3f}")
print(f"{'Significant' if chi2 > 3.84 else 'Not significant'} difference (α=0.05)")
