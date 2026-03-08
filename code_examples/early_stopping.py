"""Early Stopping — halt training when validation loss stops improving."""

import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0

    def check(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False  # keep training
        self.counter += 1
        return self.counter >= self.patience  # stop if patience exceeded

# --- demo ---
np.random.seed(42)

# simulate training with overfitting
n_train, n_val = 80, 20
X = np.random.randn(100, 3)
y = X @ np.array([2, -1, 0.5]) + np.random.randn(100) * 0.5
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

w = np.zeros(3)
stopper = EarlyStopping(patience=5)

print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'Status':>10}")
print("-" * 42)
for epoch in range(100):
    # overfit by using high LR and many epochs
    pred_t = X_train @ w
    grad = -2 * X_train.T @ (y_train - pred_t) / n_train
    w -= 0.05 * grad
    # add noise to w to simulate overfitting
    if epoch > 15:
        w += np.random.randn(3) * 0.01

    train_loss = np.mean((y_train - X_train @ w)**2)
    val_loss = np.mean((y_val - X_val @ w)**2)

    if epoch % 5 == 0 or stopper.check(val_loss, epoch):
        status = "best" if val_loss <= stopper.best_loss else ""
        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} {status:>10}")

    if stopper.check(val_loss, epoch):
        print(f"\nStopped at epoch {epoch}. Best epoch: {stopper.best_epoch} (val_loss={stopper.best_loss:.4f})")
        break
