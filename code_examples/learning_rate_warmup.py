"""Learning Rate Warmup — gradual LR increase for stable early training."""

import numpy as np

def constant_lr(step, base_lr=0.01, **kw):
    return base_lr

def linear_warmup(step, base_lr=0.01, warmup_steps=100, **kw):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr

def cosine_decay(step, base_lr=0.01, total_steps=1000, **kw):
    return base_lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))

def warmup_cosine(step, base_lr=0.01, warmup_steps=100, total_steps=1000, **kw):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def train_with_schedule(X, y, schedule_fn, total_steps=500, **kw):
    W = np.random.randn(X.shape[1]) * 0.5
    losses = []
    for step in range(total_steps):
        lr = schedule_fn(step, total_steps=total_steps, **kw)
        pred = X @ W
        loss = np.mean((pred - y)**2)
        grad = 2 * X.T @ (pred - y) / len(X)
        W -= lr * grad
        losses.append(loss)
    return losses

# --- demo ---
np.random.seed(42)
X = np.random.randn(200, 10)
w_true = np.random.randn(10)
y = X @ w_true + np.random.randn(200) * 0.1

print("=== Learning Rate Schedules ===\n")

schedules = [
    ("Constant", constant_lr, {"base_lr": 0.01}),
    ("Linear Warmup", linear_warmup, {"base_lr": 0.01, "warmup_steps": 50}),
    ("Cosine Decay", cosine_decay, {"base_lr": 0.01}),
    ("Warmup+Cosine", warmup_cosine, {"base_lr": 0.01, "warmup_steps": 50}),
]

print(f"{'Schedule':>16} {'Step 0':>8} {'Step 50':>8} {'Step 250':>8} {'Step 499':>8} {'Final Loss':>11}")
print("-" * 60)
for name, fn, kw in schedules:
    losses = train_with_schedule(X, y, fn, total_steps=500, **kw)
    lr0 = fn(0, total_steps=500, **kw)
    lr50 = fn(50, total_steps=500, **kw)
    lr250 = fn(250, total_steps=500, **kw)
    lr499 = fn(499, total_steps=500, **kw)
    print(f"{name:>16} {lr0:>8.5f} {lr50:>8.5f} {lr250:>8.5f} {lr499:>8.5f} {losses[-1]:>11.6f}")
