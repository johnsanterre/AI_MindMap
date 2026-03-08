"""Kalman Filter — 1D tracking with noisy measurements."""

import numpy as np

class KalmanFilter1D:
    def __init__(self, x0, P0, Q, R, F=1.0, H=1.0):
        self.x = x0     # state estimate
        self.P = P0     # estimate covariance
        self.Q = Q      # process noise
        self.R = R      # measurement noise
        self.F = F      # state transition
        self.H = H      # observation model

    def predict(self):
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        return self.x

    def update(self, z):
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        return self.x, K

# --- demo: track a moving object ---
np.random.seed(42)
n_steps = 50
true_pos = np.cumsum(np.ones(n_steps) * 0.5)  # constant velocity
measurements = true_pos + np.random.randn(n_steps) * 2.0  # noisy sensor

kf = KalmanFilter1D(x0=0, P0=10, Q=0.1, R=4.0)

estimates = []
for z in measurements:
    kf.predict()
    est, K = kf.update(z)
    estimates.append(est)

estimates = np.array(estimates)
mse_raw = np.mean((measurements - true_pos)**2)
mse_filtered = np.mean((estimates - true_pos)**2)

print("Kalman Filter — 1D Tracking")
print(f"  Measurement MSE: {mse_raw:.3f}")
print(f"  Filtered MSE:    {mse_filtered:.3f}")
print(f"  Improvement:     {(1 - mse_filtered/mse_raw)*100:.0f}%")
print(f"\nSample (true, measured, filtered):")
for i in range(0, 50, 10):
    print(f"  t={i:2d}: true={true_pos[i]:.1f}, meas={measurements[i]:.1f}, est={estimates[i]:.1f}")
