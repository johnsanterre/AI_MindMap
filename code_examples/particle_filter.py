"""Particle Filter — sequential Monte Carlo for nonlinear state estimation."""

import numpy as np

def particle_filter(observations, n_particles=500, process_noise=1.0, obs_noise=1.0):
    """Bootstrap particle filter for 1D state estimation."""
    T = len(observations)
    particles = np.random.randn(n_particles) * 5  # prior
    weights = np.ones(n_particles) / n_particles

    estimates = []
    for t in range(T):
        # predict: propagate particles through nonlinear dynamics
        particles = 0.5 * particles + 25 * particles / (1 + particles**2) + \
                    8 * np.cos(1.2 * t) + np.random.randn(n_particles) * process_noise

        # update: weight by likelihood
        log_w = -0.5 * ((observations[t] - particles**2 / 20)**2) / obs_noise**2
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        # estimate
        estimates.append(np.sum(weights * particles))

        # resample (systematic)
        cumsum = np.cumsum(weights)
        u = (np.arange(n_particles) + np.random.rand()) / n_particles
        indices = np.searchsorted(cumsum, u)
        particles = particles[indices]
        weights = np.ones(n_particles) / n_particles

    return np.array(estimates)

# --- demo ---
np.random.seed(42)

# generate ground truth
T = 30
states = np.zeros(T)
obs = np.zeros(T)
states[0] = 0.1
for t in range(1, T):
    states[t] = 0.5 * states[t-1] + 25 * states[t-1] / (1 + states[t-1]**2) + \
                8 * np.cos(1.2 * t) + np.random.randn() * 1.0
    obs[t] = states[t]**2 / 20 + np.random.randn() * 1.0

estimates = particle_filter(obs, n_particles=1000)

rmse = np.sqrt(np.mean((states - estimates)**2))
print("=== Particle Filter (Sequential Monte Carlo) ===")
print(f"Time steps: {T}, Particles: 1000\n")
print(f"{'t':>4} {'True State':>12} {'Estimate':>12} {'Error':>10}")
print("-" * 42)
for t in range(0, T, 3):
    err = abs(states[t] - estimates[t])
    print(f"{t:>4} {states[t]:>12.3f} {estimates[t]:>12.3f} {err:>10.3f}")
print(f"\nRMSE: {rmse:.3f}")
