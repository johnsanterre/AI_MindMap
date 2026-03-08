"""Game Theory — Nash equilibrium and strategic decision-making."""

import numpy as np

def find_nash_mixed(A, B):
    """Find mixed strategy Nash equilibrium for 2x2 games."""
    # player 2's mix makes player 1 indifferent
    # A[0,0]*q + A[0,1]*(1-q) = A[1,0]*q + A[1,1]*(1-q)
    denom = (A[0,0] - A[0,1] - A[1,0] + A[1,1])
    if abs(denom) < 1e-10:
        return None, None
    q = (A[1,1] - A[0,1]) / denom

    denom2 = (B[0,0] - B[0,1] - B[1,0] + B[1,1])
    if abs(denom2) < 1e-10:
        return None, None
    p = (B[1,1] - B[1,0]) / denom2

    if 0 <= p <= 1 and 0 <= q <= 1:
        return p, q
    return None, None

def expected_payoff(A, p, q):
    strat1 = np.array([p, 1-p])
    strat2 = np.array([q, 1-q])
    return strat1 @ A @ strat2

# --- demo ---
print("=== Game Theory: Classic Games ===\n")

# Prisoner's Dilemma
print("--- Prisoner's Dilemma ---")
# (Cooperate, Defect) payoffs
A = np.array([[-1, -3], [0, -2]])  # row player
B = np.array([[-1, 0], [-3, -2]])  # col player
print("  Payoff matrix (Row, Col):")
labels = ["Cooperate", "Defect"]
for i in range(2):
    for j in range(2):
        print(f"  ({labels[i]:>9}, {labels[j]:<9}): ({A[i,j]:>2}, {B[i,j]:>2})")
# dominant strategy
print(f"  Nash equilibrium: (Defect, Defect) with payoff ({A[1,1]}, {B[1,1]})")

# Matching Pennies (mixed strategy)
print("\n--- Matching Pennies ---")
A = np.array([[1, -1], [-1, 1]])
B = -A
p, q = find_nash_mixed(A, B)
print(f"  Row plays Heads with prob {p:.2f}")
print(f"  Col plays Heads with prob {q:.2f}")
print(f"  Expected payoff: {expected_payoff(A, p, q):.2f} (fair game)")

# Battle of the Sexes
print("\n--- Battle of the Sexes ---")
A = np.array([[3, 0], [0, 2]])
B = np.array([[2, 0], [0, 3]])
p, q = find_nash_mixed(A, B)
print(f"  Mixed Nash: p1={p:.2f}, p2={q:.2f}")
print(f"  Pure Nash: (Opera,Opera) or (Football,Football)")
print(f"  Expected payoff at mixed NE: {expected_payoff(A, p, q):.2f}")
