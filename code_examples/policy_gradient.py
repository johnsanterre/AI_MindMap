"""Policy Gradient — REINFORCE algorithm on a simple bandit."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

class MultiArmedBandit:
    def __init__(self, n_arms=5):
        self.means = np.random.randn(n_arms)

    def pull(self, arm):
        return np.random.randn() + self.means[arm]

def reinforce(env, n_episodes=2000, lr=0.1):
    n_arms = len(env.means)
    theta = np.zeros(n_arms)  # policy parameters
    reward_history = []

    for ep in range(n_episodes):
        probs = softmax(theta)
        action = np.random.choice(n_arms, p=probs)
        reward = env.pull(action)
        reward_history.append(reward)

        # policy gradient: ∇θ log π(a|θ) * R
        grad = -probs.copy()
        grad[action] += 1  # ∇θ log softmax
        theta += lr * grad * reward

    return theta, reward_history

# --- demo ---
np.random.seed(42)
env = MultiArmedBandit(5)
print(f"True arm means: {env.means.round(3)}")
print(f"Best arm: {env.means.argmax()} (mean={env.means.max():.3f})\n")

theta, rewards = reinforce(env, n_episodes=3000, lr=0.1)
probs = softmax(theta)

print("Learned policy:")
for i in range(len(env.means)):
    bar = "█" * int(probs[i] * 50)
    print(f"  Arm {i}: p={probs[i]:.3f} {bar}")

print(f"\nAvg reward (first 100):  {np.mean(rewards[:100]):.3f}")
print(f"Avg reward (last 100):   {np.mean(rewards[-100:]):.3f}")
print(f"Optimal arm selected:    {probs.argmax()} (true best: {env.means.argmax()})")
