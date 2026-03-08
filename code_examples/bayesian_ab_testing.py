"""Bayesian A/B Testing — comparing variants with posterior distributions."""

import numpy as np

def beta_posterior(successes, trials, prior_a=1, prior_b=1):
    """Beta posterior for conversion rate."""
    return prior_a + successes, prior_b + (trials - successes)

def prob_b_beats_a(a_params, b_params, n_samples=100000):
    """Monte Carlo estimate of P(B > A)."""
    samples_a = np.random.beta(a_params[0], a_params[1], n_samples)
    samples_b = np.random.beta(b_params[0], b_params[1], n_samples)
    return np.mean(samples_b > samples_a)

def expected_loss(a_params, b_params, n_samples=100000):
    """Expected loss of choosing B over A (and vice versa)."""
    samples_a = np.random.beta(a_params[0], a_params[1], n_samples)
    samples_b = np.random.beta(b_params[0], b_params[1], n_samples)
    loss_choosing_b = np.mean(np.maximum(samples_a - samples_b, 0))
    loss_choosing_a = np.mean(np.maximum(samples_b - samples_a, 0))
    return loss_choosing_a, loss_choosing_b

def credible_interval(alpha, beta_param, ci=0.95):
    """HDI approximation using quantiles."""
    samples = np.random.beta(alpha, beta_param, 100000)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return lo, hi

# --- demo ---
np.random.seed(42)

print("=== Bayesian A/B Testing ===\n")

# scenario: website conversion rates
tests = [
    ("Small sample", 10, 100, 15, 100),
    ("Medium sample", 50, 500, 65, 500),
    ("Large sample", 500, 5000, 550, 5000),
    ("Close rates", 100, 1000, 105, 1000),
]

for name, conv_a, n_a, conv_b, n_b in tests:
    a_params = beta_posterior(conv_a, n_a)
    b_params = beta_posterior(conv_b, n_b)
    p_b_wins = prob_b_beats_a(a_params, b_params)
    loss_a, loss_b = expected_loss(a_params, b_params)
    ci_a = credible_interval(*a_params)
    ci_b = credible_interval(*b_params)

    print(f"--- {name} ---")
    print(f"  A: {conv_a}/{n_a} = {conv_a/n_a:.1%}, 95% CI: ({ci_a[0]:.3f}, {ci_a[1]:.3f})")
    print(f"  B: {conv_b}/{n_b} = {conv_b/n_b:.1%}, 95% CI: ({ci_b[0]:.3f}, {ci_b[1]:.3f})")
    print(f"  P(B > A) = {p_b_wins:.1%}")
    print(f"  Expected loss: choosing A={loss_a:.4f}, choosing B={loss_b:.4f}")
    decision = "Choose B" if p_b_wins > 0.95 else "Choose A" if p_b_wins < 0.05 else "Need more data"
    print(f"  Decision: {decision}\n")
