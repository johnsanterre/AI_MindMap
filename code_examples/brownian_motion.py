"""Brownian Motion — simulation and properties."""

import numpy as np

def brownian_motion(n_steps, dt=0.01, n_paths=5):
    """Simulate standard Brownian motion paths."""
    dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    W = np.cumsum(dW, axis=1)
    W = np.column_stack([np.zeros(n_paths), W])  # start at 0
    return W

def geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths=5):
    """Stock price simulation: dS = μS dt + σS dW."""
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)
    W = brownian_motion(n_steps, dt, n_paths)
    S = S0 * np.exp((mu - 0.5*sigma**2) * t + sigma * W)
    return t, S

def check_properties(W, dt):
    """Verify Brownian motion properties."""
    n_paths, n_steps = W.shape[0], W.shape[1] - 1
    increments = np.diff(W, axis=1)
    print(f"  E[W(T)]: {W[:, -1].mean():.4f} (expected: 0)")
    print(f"  Var[W(T)]: {W[:, -1].var():.4f} (expected: {n_steps*dt:.4f})")
    print(f"  E[increment]: {increments.mean():.6f} (expected: 0)")
    print(f"  Var[increment]: {increments.var():.6f} (expected: {dt:.6f})")

# --- demo ---
np.random.seed(42)

# standard Brownian motion
print("=== Standard Brownian Motion ===")
W = brownian_motion(1000, dt=0.001, n_paths=10000)
check_properties(W, 0.001)
print(f"  Final positions (5 paths): {W[:5, -1].round(3)}")

# geometric Brownian motion (stock simulation)
print("\n=== Geometric Brownian Motion (Stock Price) ===")
t, S = geometric_brownian_motion(S0=100, mu=0.1, sigma=0.3, T=1.0, n_steps=252, n_paths=10000)
print(f"  Initial price: $100")
print(f"  Mean final price: ${S[:, -1].mean():.2f}")
print(f"  Median final price: ${np.median(S[:, -1]):.2f}")
print(f"  5th percentile: ${np.percentile(S[:, -1], 5):.2f}")
print(f"  95th percentile: ${np.percentile(S[:, -1], 95):.2f}")
print(f"  P(price > 100): {(S[:, -1] > 100).mean():.1%}")
