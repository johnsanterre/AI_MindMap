"""Learning Rate Scheduling — step decay, exponential, cosine annealing, warmup."""

import numpy as np

def step_decay(epoch, lr0=0.1, drop=0.5, every=10):
    return lr0 * (drop ** (epoch // every))

def exponential_decay(epoch, lr0=0.1, gamma=0.95):
    return lr0 * (gamma ** epoch)

def cosine_annealing(epoch, lr0=0.1, T_max=50):
    return lr0 * 0.5 * (1 + np.cos(np.pi * epoch / T_max))

def warmup_cosine(epoch, lr0=0.1, warmup=5, T_max=50):
    if epoch < warmup:
        return lr0 * epoch / warmup
    return lr0 * 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (T_max - warmup)))

# --- demo ---
epochs = 50
print(f"{'Epoch':>6} {'Step':>8} {'Exp':>8} {'Cosine':>8} {'Warmup+Cos':>10}")
print("-" * 46)
for e in range(0, epochs, 5):
    s = step_decay(e)
    ex = exponential_decay(e)
    c = cosine_annealing(e, T_max=epochs)
    w = warmup_cosine(e, T_max=epochs)
    print(f"{e:>6} {s:>8.5f} {ex:>8.5f} {c:>8.5f} {w:>10.5f}")
