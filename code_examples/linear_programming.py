"""Linear Programming — simplex-like method for optimization with constraints."""

import numpy as np

def simple_lp(c, A_ub, b_ub):
    """Solve min c^T x s.t. Ax <= b, x >= 0 using basic feasible point search."""
    n_constraints, n_vars = A_ub.shape

    # add slack variables: [A | I] [x; s] = b
    A_aug = np.hstack([A_ub, np.eye(n_constraints)])
    c_aug = np.concatenate([c, np.zeros(n_constraints)])
    basis = list(range(n_vars, n_vars + n_constraints))
    n_total = n_vars + n_constraints

    for _ in range(100):
        # compute reduced costs
        B = A_aug[:, basis]
        B_inv = np.linalg.inv(B)
        cb = c_aug[basis]
        x_B = B_inv @ b_ub

        # find entering variable
        reduced = c_aug - cb @ B_inv @ A_aug
        entering = -1
        for j in range(n_total):
            if j not in basis and reduced[j] < -1e-8:
                entering = j
                break
        if entering == -1:
            break  # optimal

        # find leaving variable (minimum ratio test)
        d = B_inv @ A_aug[:, entering]
        ratios = []
        for i in range(n_constraints):
            if d[i] > 1e-8:
                ratios.append((x_B[i] / d[i], i))
        if not ratios:
            return None, None  # unbounded
        _, leaving_idx = min(ratios)
        basis[leaving_idx] = entering

    x = np.zeros(n_total)
    x[basis] = B_inv @ b_ub
    return x[:n_vars], c @ x[:n_vars]

# --- demo ---
print("=== Linear Programming ===\n")

# maximize 3x + 2y → minimize -3x - 2y
# s.t. x + y <= 4, x + 3y <= 6, x,y >= 0
c = np.array([-3.0, -2.0])
A = np.array([[1, 1], [1, 3]])
b = np.array([4.0, 6.0])

x_opt, val = simple_lp(c, A, b)
print("Problem: max 3x + 2y")
print("  s.t. x + y ≤ 4")
print("       x + 3y ≤ 6")
print("       x, y ≥ 0\n")
print(f"Optimal: x={x_opt[0]:.2f}, y={x_opt[1]:.2f}")
print(f"Max value: {-val:.2f}")

# production planning
print("\n--- Production Planning ---")
# 2 products, 3 resources
c2 = np.array([-5.0, -4.0])  # profit
A2 = np.array([[6, 4], [1, 2], [2, 1]])  # resource usage
b2 = np.array([24.0, 6.0, 8.0])  # available resources
x2, v2 = simple_lp(c2, A2, b2)
print(f"Product A: {x2[0]:.1f} units, Product B: {x2[1]:.1f} units")
print(f"Max profit: ${-v2:.1f}")
