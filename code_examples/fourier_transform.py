"""Fourier Transform — DFT from scratch and frequency analysis."""

import numpy as np

def dft(x):
    """Discrete Fourier Transform (O(n²) — educational, not FFT)."""
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x

def idft(X):
    """Inverse DFT."""
    N = len(X)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(2j * np.pi * k * n / N)
    return (W @ X) / N

def frequency_spectrum(x, sample_rate=1.0):
    """Get frequency and magnitude from DFT."""
    X = dft(x)
    N = len(x)
    freqs = np.arange(N) * sample_rate / N
    magnitudes = np.abs(X) / N
    return freqs[:N//2], magnitudes[:N//2] * 2

# --- demo ---
np.random.seed(42)
N = 128
t = np.arange(N) / N

# signal = 3Hz + 7Hz + noise
signal = 2 * np.sin(2*np.pi*3*t) + 0.5 * np.sin(2*np.pi*7*t) + np.random.randn(N) * 0.3

freqs, mags = frequency_spectrum(signal, sample_rate=N)

print("Fourier Transform — Frequency Detection")
print(f"Signal: 2·sin(3·2πt) + 0.5·sin(7·2πt) + noise\n")

# find peaks
top_idx = mags.argsort()[-5:][::-1]
print("Top frequencies by magnitude:")
for i in top_idx:
    if mags[i] > 0.1:
        print(f"  f={freqs[i]:.1f} Hz, magnitude={mags[i]:.3f}")

# verify round-trip
X = dft(signal)
recovered = idft(X).real
print(f"\nRound-trip error: {np.max(np.abs(signal - recovered)):.2e}")
