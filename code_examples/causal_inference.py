"""Causal Inference — estimating treatment effects from observational data."""

import numpy as np

def ate_naive(y_treated, y_control):
    """Naive Average Treatment Effect."""
    return y_treated.mean() - y_control.mean()

def ate_regression_adjustment(X, y, treatment):
    """Control for confounders via linear regression."""
    # fit y ~ treatment + X
    n = len(y)
    design = np.column_stack([np.ones(n), treatment, X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta[1]  # coefficient of treatment

def propensity_score(X, treatment, lr=0.1, n_iter=200):
    """Logistic regression for P(treatment | X)."""
    n, d = X.shape
    w = np.zeros(d + 1)
    X_aug = np.column_stack([np.ones(n), X])
    for _ in range(n_iter):
        z = X_aug @ w
        p = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
        grad = X_aug.T @ (p - treatment) / n
        w -= lr * grad
    return 1 / (1 + np.exp(-np.clip(X_aug @ w, -10, 10)))

def ipw_ate(y, treatment, ps):
    """Inverse Propensity Weighting ATE."""
    treated = treatment == 1
    control = treatment == 0
    ate = np.mean(y[treated] / ps[treated]) - np.mean(y[control] / (1 - ps[control]))
    return ate / len(y) * (treated.sum() + control.sum())  # normalize

# --- demo ---
np.random.seed(42)
n = 500

# confounder: age affects both treatment assignment and outcome
age = np.random.normal(50, 10, n)
X = np.column_stack([age, np.random.randn(n)])

# treatment more likely for younger people (confounded!)
p_treat = 1 / (1 + np.exp(0.1 * (age - 50)))
treatment = (np.random.rand(n) < p_treat).astype(float)

# true treatment effect = 5, but age also affects outcome
true_ate = 5.0
y = 0.5 * age + true_ate * treatment + np.random.randn(n) * 3

print("=== Causal Inference ===")
print(f"True ATE: {true_ate}")
print(f"Treated: {int(treatment.sum())}, Control: {int((1-treatment).sum())}\n")

naive = ate_naive(y[treatment == 1], y[treatment == 0])
print(f"Naive ATE:              {naive:.2f} (biased by age confounding)")

reg_ate = ate_regression_adjustment(X, y, treatment)
print(f"Regression adjustment:  {reg_ate:.2f}")

ps = propensity_score(X, treatment)
ipw = ipw_ate(y, treatment, ps)
print(f"IPW ATE:                {ipw:.2f}")

print(f"\nNaive estimate is biased because younger people (lower outcome)")
print(f"are more likely to receive treatment. Adjustment corrects this.")
