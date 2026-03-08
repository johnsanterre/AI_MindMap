"""Weight Decay — L2 regularization applied directly to weights."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def train(X, y_oh, lr=0.1, wd=0.0, n_epochs=200):
    n, d = X.shape
    k = y_oh.shape[1]
    W = np.random.randn(d, k) * 0.1
    losses = []
    for _ in range(n_epochs):
        logits = X @ W
        probs = softmax(logits)
        loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-8), axis=1))
        loss += 0.5 * wd * np.sum(W**2)  # L2 penalty
        grad = X.T @ (probs - y_oh) / n + wd * W
        W -= lr * grad
        losses.append(loss)
    return W, losses

# --- demo ---
np.random.seed(42)

# overfit-prone: many features, few samples
n_train, n_test = 30, 100
d = 50
n_classes = 3
X_all = np.random.randn(n_train + n_test, d)
true_w = np.zeros(d); true_w[:5] = np.array([2, -1.5, 1, -0.5, 0.8])
y_all = (X_all @ true_w + np.random.randn(n_train + n_test) * 0.5)
y_all = np.digitize(y_all, bins=np.percentile(y_all, [33, 66])).clip(0, 2)
y_oh = np.eye(n_classes)[y_all]

X_tr, X_te = X_all[:n_train], X_all[n_train:]
y_tr, y_te = y_oh[:n_train], y_oh[n_train:]

print("=== Weight Decay (L2 Regularization) ===")
print(f"Train: {n_train}, Test: {n_test}, Features: {d}\n")
print(f"{'λ':>8} {'Train Loss':>11} {'Test Acc':>10} {'||W||':>8}")
print("-" * 42)
for wd in [0.0, 0.001, 0.01, 0.1, 1.0]:
    W, losses = train(X_tr, y_tr, lr=0.1, wd=wd)
    train_pred = softmax(X_tr @ W).argmax(axis=1)
    test_pred = softmax(X_te @ W).argmax(axis=1)
    test_acc = (test_pred == y_te.argmax(axis=1)).mean()
    w_norm = np.linalg.norm(W)
    print(f"{wd:>8.3f} {losses[-1]:>11.4f} {test_acc:>10.1%} {w_norm:>8.3f}")

print("\nWeight decay shrinks weights, reducing overfitting.")
