"""Q-Learning — tabular reinforcement learning on a grid world."""

import numpy as np

class GridWorld:
    """4x4 grid. Start top-left, goal bottom-right. -1 per step, +10 at goal."""
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        r, c = self.pos
        if action == 0: r = max(0, r-1)        # up
        elif action == 1: r = min(self.size-1, r+1)  # down
        elif action == 2: c = max(0, c-1)       # left
        elif action == 3: c = min(self.size-1, c+1)  # right
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 10.0 if done else -1.0
        return self.pos, reward, done

def q_learning(env, episodes=500, lr=0.1, gamma=0.95, epsilon=0.1):
    Q = np.zeros((env.size, env.size, 4))
    rewards_per_ep = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = Q[state].argmax()
            next_state, reward, done = env.step(action)
            Q[state][action] += lr * (reward + gamma * Q[next_state].max() - Q[state][action])
            state = next_state
            total_reward += reward
            if done:
                break
        rewards_per_ep.append(total_reward)
    return Q, rewards_per_ep

# --- demo ---
np.random.seed(42)
env = GridWorld(4)
Q, rewards = q_learning(env, episodes=1000)

print("Learned policy (arrows show best action):")
arrows = ["↑", "↓", "←", "→"]
for r in range(4):
    row = ""
    for c in range(4):
        if (r, c) == env.goal:
            row += "  ★ "
        else:
            row += f"  {arrows[Q[r, c].argmax()]} "
    print(row)

print(f"\nAvg reward (last 100 episodes): {np.mean(rewards[-100:]):.1f}")
print(f"Avg reward (first 100 episodes): {np.mean(rewards[:100]):.1f}")
