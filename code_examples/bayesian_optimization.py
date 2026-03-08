"""Bayesian Optimization — sample-efficient global optimization with GP surrogate."""

import numpy as np

def rbf_kernel(x1, x2, l=1.0):
    return np.exp(-0.5 * (x1[:, None] - x2[None, :])**2 / l**2)

def gp_predict(X_train, y_train, X_test, l=1.0, noise=1e-6):
    K = rbf_kernel(X_train, X_train, l) + noise * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, l)
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    K_ss = rbf_kernel(X_test, X_test, l)
    cov = K_ss - K_s.T @ K_inv @ K_s
    std = np.sqrt(np.diag(cov).clip(0))
    return mu, std

def expected_improvement(mu, std, y_best, xi=0.01):
    z = (mu - y_best - xi) / (std + 1e-8)
    # approximate EI using standard normal
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + np.vectorize(lambda x: np.tanh(x * 0.7978))(z / np.sqrt(2)))
    return (mu - y_best - xi) * Phi + std * phi

def bayesian_optimize(f, bounds, n_init=3, n_iter=15):
    # initial random samples
    X = np.random.uniform(bounds[0], bounds[1], n_init)
    y = np.array([f(x) for x in X])
    history = [(X.copy(), y.copy())]

    X_candidates = np.linspace(bounds[0], bounds[1], 200)

    for _ in range(n_iter):
        mu, std = gp_predict(X, y, X_candidates, l=0.5)
        ei = expected_improvement(mu, std, y.max())
        next_x = X_candidates[ei.argmax()]
        next_y = f(next_x)
        X = np.append(X, next_x)
        y = np.append(y, next_y)
        history.append((X.copy(), y.copy()))

    return X[y.argmax()], y.max(), history

# --- demo ---
np.random.seed(42)

# objective with multiple peaks
def objective(x):
    return np.sin(3*x) * np.exp(-0.1*x**2) + 0.5*np.cos(7*x) * np.exp(-0.5*(x-1)**2)

x_best, y_best, hist = bayesian_optimize(objective, bounds=(-3, 3), n_init=3, n_iter=12)

print("=== Bayesian Optimization ===")
print(f"Maximizing f(x) = sin(3x)e^(-0.1x²) + 0.5cos(7x)e^(-0.5(x-1)²)\n")

# true optimum (by dense search)
x_dense = np.linspace(-3, 3, 10000)
y_dense = np.array([objective(x) for x in x_dense])
true_best = x_dense[y_dense.argmax()]

print(f"{'Eval':>5} {'x':>8} {'f(x)':>8} {'Best so far':>12}")
print("-" * 36)
X_all, y_all = hist[-1]
for i in range(len(X_all)):
    best_so_far = y_all[:i+1].max()
    print(f"{i+1:>5} {X_all[i]:>8.4f} {y_all[i]:>8.4f} {best_so_far:>12.4f}")

print(f"\nBO found:   x={x_best:.4f}, f={y_best:.4f}")
print(f"True best:  x={true_best:.4f}, f={y_dense.max():.4f}")
print(f"Found optimum in {len(X_all)} evaluations (vs 10000 grid search).")
