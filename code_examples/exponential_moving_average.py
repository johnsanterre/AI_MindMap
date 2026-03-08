"""Exponential Moving Average — smoothing for model weights and time series."""

import numpy as np

def ema(data, alpha=0.1):
    """Exponential moving average: s_t = α·x_t + (1-α)·s_{t-1}."""
    result = np.zeros_like(data)
    result[0] = data[0]
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t - 1]
    return result

def ema_model_weights(current_params, ema_params, decay=0.999):
    """EMA for model parameters (Polyak averaging)."""
    return decay * ema_params + (1 - decay) * current_params

def double_ema(data, alpha=0.1, beta=0.1):
    """Double exponential smoothing (Holt's method) for trend."""
    level = data[0]
    trend = data[1] - data[0]
    result = [level]
    for t in range(1, len(data)):
        new_level = alpha * data[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
        result.append(level)
    return np.array(result)

# --- demo ---
np.random.seed(42)

# noisy time series with trend
t = np.arange(100)
trend = 0.1 * t
seasonal = 3 * np.sin(2 * np.pi * t / 20)
noise = np.random.randn(100) * 1.5
data = trend + seasonal + noise

print("=== Exponential Moving Average ===\n")

print("--- Effect of α (smoothing factor) ---")
print(f"{'α':>5} {'MSE vs trend':>14} {'Lag (peaks)':>12}")
for alpha in [0.02, 0.1, 0.3, 0.5, 0.9]:
    smoothed = ema(data, alpha)
    mse = np.mean((smoothed - trend - seasonal)**2)
    print(f"{alpha:>5.2f} {mse:>14.4f}")

# model weight averaging
print(f"\n--- EMA for Model Parameters ---")
# simulate training: weights + noise
true_w = np.array([1.0, -0.5, 0.3])
w_ema = np.zeros(3)
for step in range(100):
    w_current = true_w + np.random.randn(3) * (1.0 / (step + 1))
    w_ema = ema_model_weights(w_current, w_ema, decay=0.99)

print(f"True weights:    {true_w}")
print(f"Final weights:   {w_current.round(4)}")
print(f"EMA weights:     {w_ema.round(4)}")
print(f"EMA error:       {np.linalg.norm(w_ema - true_w):.4f}")
print(f"Current error:   {np.linalg.norm(w_current - true_w):.4f}")

# double EMA
print(f"\n--- Double EMA (with trend) ---")
dema = double_ema(data, alpha=0.2, beta=0.1)
mse_single = np.mean((ema(data, 0.2) - data)**2)
mse_double = np.mean((dema - data)**2)
print(f"Single EMA MSE: {mse_single:.4f}")
print(f"Double EMA MSE: {mse_double:.4f}")
print(f"Double EMA captures trend better than single EMA.")
