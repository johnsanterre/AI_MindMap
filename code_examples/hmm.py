"""Hidden Markov Model — forward algorithm and Viterbi decoding."""

import numpy as np

def forward_algorithm(obs, A, B, pi):
    """P(observations | model) via forward algorithm."""
    T = len(obs)
    N = len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
    return alpha, alpha[-1].sum()

def viterbi(obs, A, B, pi):
    """Most likely state sequence via Viterbi."""
    T = len(obs)
    N = len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            probs = delta[t-1] * A[:, j]
            psi[t, j] = probs.argmax()
            delta[t, j] = probs.max() * B[j, obs[t]]
    # backtrack
    path = [delta[-1].argmax()]
    for t in range(T-1, 0, -1):
        path.append(psi[t, path[-1]])
    return list(reversed(path))

# --- demo: dishonest casino ---
np.random.seed(42)
states = ["Fair", "Loaded"]
symbols = [1, 2, 3, 4, 5, 6]

pi = np.array([0.5, 0.5])
A = np.array([[0.95, 0.05],   # Fair→Fair, Fair→Loaded
              [0.10, 0.90]])   # Loaded→Fair, Loaded→Loaded
B = np.array([[1/6]*6,                        # Fair die
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])  # Loaded die (favors 6)

# generate observations
true_states = [0]
for _ in range(29):
    true_states.append(np.random.choice(2, p=A[true_states[-1]]))
obs = [np.random.choice(6, p=B[s]) for s in true_states]

# decode
alpha, prob = forward_algorithm(obs, A, B, pi)
decoded = viterbi(obs, A, B, pi)

print("Dice rolls:", [symbols[o] for o in obs])
print("True state:", ['F' if s==0 else 'L' for s in true_states])
print("Viterbi:   ", ['F' if s==0 else 'L' for s in decoded])
print(f"Accuracy:   {sum(a==b for a,b in zip(true_states, decoded))/len(obs):.1%}")
print(f"P(obs|model): {prob:.2e}")
