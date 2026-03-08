"""Multi-Armed Bandit — exploration vs exploitation strategies."""

import numpy as np

class Bandit:
    def __init__(self, true_means):
        self.means = true_means
        self.k = len(true_means)

    def pull(self, arm):
        return np.random.randn() + self.means[arm]

def epsilon_greedy(bandit, n_steps=1000, epsilon=0.1):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []
    for _ in range(n_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(bandit.k)
        else:
            a = Q.argmax()
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
    return np.array(rewards)

def ucb(bandit, n_steps=1000, c=2.0):
    Q = np.zeros(bandit.k)
    N = np.zeros(bandit.k)
    rewards = []
    for t in range(n_steps):
        if t < bandit.k:
            a = t
        else:
            ucb_vals = Q + c * np.sqrt(np.log(t) / (N + 1e-8))
            a = ucb_vals.argmax()
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards.append(r)
    return np.array(rewards)

def thompson_sampling(bandit, n_steps=1000):
    alpha = np.ones(bandit.k)
    beta = np.ones(bandit.k)
    rewards = []
    for _ in range(n_steps):
        samples = np.random.beta(alpha, beta)
        a = samples.argmax()
        r = bandit.pull(a)
        if r > 0:
            alpha[a] += 1
        else:
            beta[a] += 1
        rewards.append(r)
    return np.array(rewards)

# --- demo ---
np.random.seed(42)
means = [0.2, 0.5, 0.8, 0.3, 0.6]
bandit = Bandit(means)

n = 2000
print(f"=== Multi-Armed Bandit ({len(means)} arms) ===")
print(f"True means: {means}\n")

print(f"{'Strategy':>20} {'Avg Reward':>12} {'Best Arm %':>12}")
print("-" * 48)

for name, fn in [("ε-greedy (0.1)", lambda: epsilon_greedy(bandit, n, 0.1)),
                  ("ε-greedy (0.01)", lambda: epsilon_greedy(bandit, n, 0.01)),
                  ("UCB (c=2)", lambda: ucb(bandit, n)),
                  ("Thompson", lambda: thompson_sampling(bandit, n))]:
    rewards = fn()
    avg = rewards.mean()
    # regret: best possible - actual
    regret = max(means) * n - rewards.sum()
    print(f"{name:>20} {avg:>12.3f} {regret:>12.1f} regret")
