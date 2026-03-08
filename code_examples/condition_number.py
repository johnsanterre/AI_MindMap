"""Condition Number — measuring numerical sensitivity of linear systems."""

import numpy as np

def condition_number(A):
    """κ(A) = ||A|| · ||A⁻¹|| = σ_max / σ_min."""
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1]

def solve_perturbed(A, b, perturbation=1e-6):
    """Show how small perturbations in b affect solution."""
    x_exact = np.linalg.solve(A, b)
    db = np.random.randn(len(b)) * perturbation
    x_perturbed = np.linalg.solve(A, b + db)
    rel_error_b = np.linalg.norm(db) / np.linalg.norm(b)
    rel_error_x = np.linalg.norm(x_perturbed - x_exact) / np.linalg.norm(x_exact)
    return rel_error_b, rel_error_x, rel_error_x / rel_error_b

# --- demo ---
np.random.seed(42)

print("=== Condition Number ===\n")

# well-conditioned system
A_good = np.array([[2, 1], [1, 3]], dtype=float)
# ill-conditioned system (Hilbert matrix)
A_bad = np.array([[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]])
# nearly singular
A_ugly = np.array([[1, 1], [1, 1.0001]], dtype=float)

print(f"{'Matrix':>15} {'κ(A)':>12} {'Digits lost':>12}")
print("-" * 42)
for name, A in [("Well-cond 2×2", A_good), ("Hilbert 3×3", A_bad), ("Near-singular", A_ugly)]:
    kappa = condition_number(A)
    digits = np.log10(kappa)
    print(f"{name:>15} {kappa:>12.1f} {digits:>12.1f}")

# perturbation analysis
print(f"\n--- Perturbation Analysis ---")
print(f"Solving Ax=b with random perturbation in b (ε=1e-6)\n")
for name, A in [("Well-cond", A_good), ("Near-singular", A_ugly)]:
    b = np.ones(A.shape[0])
    kappa = condition_number(A)
    errors = [solve_perturbed(A, b) for _ in range(100)]
    amplifications = [e[2] for e in errors]
    print(f"  {name}: κ={kappa:.1f}")
    print(f"    Max error amplification: {max(amplifications):.1f}×")
    print(f"    Avg error amplification: {np.mean(amplifications):.1f}×")
    print(f"    κ predicts worst case:   {kappa:.1f}×\n")

print("Rule: you lose ≈ log₁₀(κ) digits of accuracy.")
