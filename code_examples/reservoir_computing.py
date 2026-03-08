"""Reservoir Computing — Echo State Network for time series prediction."""

import numpy as np

class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, sparsity=0.1):
        self.n_reservoir = n_reservoir
        # input weights
        self.W_in = np.random.randn(n_reservoir, n_input) * 0.1
        # reservoir weights (sparse, scaled to spectral radius)
        W = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) < sparsity
        W *= mask
        rho = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / (rho + 1e-8))
        self.W_out = None

    def run_reservoir(self, X, washout=50):
        T = len(X)
        states = np.zeros((T, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        for t in range(T):
            state = np.tanh(self.W_in @ X[t] + self.W @ state)
            states[t] = state
        return states[washout:]

    def train(self, X, y, washout=50, reg=1e-6):
        states = self.run_reservoir(X, washout)
        y_train = y[washout:]
        # ridge regression
        self.W_out = np.linalg.solve(
            states.T @ states + reg * np.eye(self.n_reservoir),
            states.T @ y_train
        )
        return states @ self.W_out

    def predict(self, X, washout=0):
        states = self.run_reservoir(X, washout)
        return states @ self.W_out

# --- demo ---
np.random.seed(42)

# Mackey-Glass-like chaotic series
T = 500
x = np.zeros(T)
x[0] = 0.9
for t in range(1, T):
    x[t] = x[t-1] + 0.2 * x[t-1] / (1 + x[t-1]**10) * 0.1 - 0.1 * x[t-1] + np.random.randn() * 0.01

X = x[:-1].reshape(-1, 1)
y = x[1:]

# train/test split
split = 350
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

esn = EchoStateNetwork(1, 200, 1, spectral_radius=0.95, sparsity=0.1)
y_pred_train = esn.train(X_train, y_train, washout=50)
y_pred_test = esn.predict(X_test)

train_rmse = np.sqrt(np.mean((y_train[50:] - y_pred_train)**2))
test_rmse = np.sqrt(np.mean((y_test - y_pred_test)**2))

print("=== Echo State Network (Reservoir Computing) ===")
print(f"Reservoir size: 200, Spectral radius: 0.95")
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE:  {test_rmse:.6f}")
print(f"\nSample predictions (test set):")
print(f"{'t':>5} {'Actual':>10} {'Predicted':>10}")
for i in range(0, min(20, len(y_test)), 2):
    print(f"{i:>5} {y_test[i]:>10.4f} {y_pred_test[i]:>10.4f}")
