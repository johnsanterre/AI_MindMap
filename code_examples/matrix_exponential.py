"""Matrix Exponential — computing e^A for differential equations and Markov chains."""

import numpy as np

def matrix_exp_taylor(A, n_terms=20):
    """e^A via Taylor series: sum A^k / k!."""
    result = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, n_terms):
        term = term @ A / k
        result += term
    return result

def matrix_exp_pade(A, order=6):
    """Padé approximation for e^A."""
    n = A.shape[0]
    I = np.eye(n)
    A2 = A @ A
    if order == 6:
        U = A @ (A2 @ (A2 + 420*I) + 15120*I)
        V = A2 @ (A2 + 420*I) + 15120*I + A2 @ 1680
        # simplified: use scaling and squaring
    return matrix_exp_taylor(A)  # fallback to Taylor for simplicity

def solve_linear_ode(A, x0, t):
    """dx/dt = Ax → x(t) = e^{At} x(0)."""
    return matrix_exp_taylor(A * t) @ x0

# --- demo ---
np.random.seed(42)

print("=== Matrix Exponential ===\n")

# simple 2x2
A = np.array([[0, -1], [1, 0]], dtype=float)  # rotation
print("A (rotation):")
print(A)
expA = matrix_exp_taylor(A)
expA_np = np.linalg.matrix_power(np.eye(2), 1)  # for reference

print(f"\ne^A (Taylor):\n{expA.round(6)}")
print(f"This is rotation by 1 radian: cos(1)={np.cos(1):.6f}, sin(1)={np.sin(1):.6f}")

# property: e^0 = I
print(f"\ne^0 = I? {np.allclose(matrix_exp_taylor(np.zeros((2,2))), np.eye(2))}")

# Markov chain steady state
print(f"\n--- Continuous-time Markov Chain ---")
Q = np.array([[-0.5, 0.5, 0.0],
              [0.3, -0.7, 0.4],
              [0.1, 0.2, -0.3]])
p0 = np.array([1.0, 0, 0])

print(f"Rate matrix Q:\n{Q}")
print(f"\nState probabilities p(t) = p(0) @ e^(Qt):")
print(f"{'t':>6}", end="")
for s in range(3):
    print(f"{'State '+str(s):>10}", end="")
print()
for t in [0, 0.5, 1, 2, 5, 20]:
    pt = p0 @ matrix_exp_taylor(Q * t)
    print(f"{t:>6.1f}", end="")
    for s in range(3):
        print(f"{pt[s]:>10.4f}", end="")
    print()

# solve ODE
print(f"\n--- Linear ODE: dx/dt = Ax ---")
A2 = np.array([[-1, 2], [0, -3]])
x0 = np.array([1.0, 1.0])
for t in [0, 0.5, 1, 2]:
    xt = solve_linear_ode(A2, x0, t)
    print(f"  x({t}) = {xt.round(4)}")
