"""Gradient Accumulation — simulating large batches with limited memory."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def train_standard(X, y_oh, batch_size, lr=0.1, n_epochs=50):
    W = np.random.randn(X.shape[1], y_oh.shape[1]) * 0.1
    n = len(X)
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch_x = X[idx[i:i+batch_size]]
            batch_y = y_oh[idx[i:i+batch_size]]
            probs = softmax(batch_x @ W)
            grad = batch_x.T @ (probs - batch_y) / len(batch_x)
            W -= lr * grad
    return W

def train_accumulated(X, y_oh, micro_batch, accum_steps, lr=0.1, n_epochs=50):
    """Accumulate gradients over multiple micro-batches."""
    W = np.random.randn(X.shape[1], y_oh.shape[1]) * 0.1
    n = len(X)
    effective_batch = micro_batch * accum_steps
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, effective_batch):
            grad_accum = np.zeros_like(W)
            for step in range(accum_steps):
                start = i + step * micro_batch
                batch_x = X[idx[start:start+micro_batch]]
                batch_y = y_oh[idx[start:start+micro_batch]]
                if len(batch_x) == 0:
                    break
                probs = softmax(batch_x @ W)
                grad_accum += batch_x.T @ (probs - batch_y) / effective_batch
            W -= lr * grad_accum
    return W

# --- demo ---
np.random.seed(42)
n, d, k = 200, 8, 3
X = np.random.randn(n, d)
y = np.random.randint(0, k, n)
y_oh = np.eye(k)[y]

print("=== Gradient Accumulation ===\n")
print(f"{'Method':>25} {'Eff. Batch':>11} {'Accuracy':>10}")
print("-" * 50)

# standard training with different batch sizes
for bs in [10, 50, 200]:
    np.random.seed(0)
    W = train_standard(X, y_oh, bs)
    acc = (softmax(X @ W).argmax(axis=1) == y).mean()
    print(f"{'Standard batch='+str(bs):>25} {bs:>11} {acc:>10.1%}")

# accumulated: micro_batch=10, accumulate 5x = effective 50
np.random.seed(0)
W = train_accumulated(X, y_oh, micro_batch=10, accum_steps=5)
acc = (softmax(X @ W).argmax(axis=1) == y).mean()
print(f"{'Accum 10×5':>25} {50:>11} {acc:>10.1%}")

np.random.seed(0)
W = train_accumulated(X, y_oh, micro_batch=10, accum_steps=20)
acc = (softmax(X @ W).argmax(axis=1) == y).mean()
print(f"{'Accum 10×20':>25} {200:>11} {acc:>10.1%}")

print("\nGradient accumulation lets you train with large effective batches")
print("even when each micro-batch must be small (limited GPU memory).")
