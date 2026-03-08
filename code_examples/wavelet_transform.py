"""Wavelet Transform — time-frequency analysis with localized basis functions."""

import numpy as np

def haar_wavelet_1d(signal):
    """One level of Haar wavelet decomposition."""
    n = len(signal)
    approx = (signal[0::2] + signal[1::2]) / np.sqrt(2)
    detail = (signal[0::2] - signal[1::2]) / np.sqrt(2)
    return approx, detail

def haar_inverse(approx, detail):
    n = len(approx) * 2
    signal = np.zeros(n)
    signal[0::2] = (approx + detail) / np.sqrt(2)
    signal[1::2] = (approx - detail) / np.sqrt(2)
    return signal

def multilevel_decompose(signal, levels):
    coeffs = []
    current = signal.copy()
    for _ in range(levels):
        approx, detail = haar_wavelet_1d(current)
        coeffs.append(detail)
        current = approx
    coeffs.append(current)  # final approximation
    return coeffs

def multilevel_reconstruct(coeffs):
    current = coeffs[-1]  # start from approximation
    for detail in reversed(coeffs[:-1]):
        current = haar_inverse(current, detail)
    return current

def denoise(signal, threshold):
    coeffs = multilevel_decompose(signal, 3)
    # soft threshold detail coefficients
    for i in range(len(coeffs) - 1):
        coeffs[i] = np.sign(coeffs[i]) * np.maximum(np.abs(coeffs[i]) - threshold, 0)
    return multilevel_reconstruct(coeffs)

# --- demo ---
np.random.seed(42)

# signal with step + sine (need length = power of 2)
n = 128
t = np.linspace(0, 1, n)
clean = np.where(t < 0.5, np.sin(2*np.pi*5*t), np.sin(2*np.pi*15*t))
noisy = clean + np.random.randn(n) * 0.3

print("=== Haar Wavelet Transform ===\n")

# single level
approx, detail = haar_wavelet_1d(noisy)
reconstructed = haar_inverse(approx, detail)
print(f"Signal length: {n}")
print(f"Approx coeffs: {len(approx)}, Detail coeffs: {len(detail)}")
print(f"Reconstruction error: {np.max(np.abs(noisy - reconstructed)):.2e}")

# multi-level
print(f"\n--- Multi-level decomposition (3 levels) ---")
coeffs = multilevel_decompose(noisy, 3)
for i, c in enumerate(coeffs[:-1]):
    print(f"  Level {i+1} detail: {len(c)} coeffs, energy={np.sum(c**2):.3f}")
print(f"  Final approx: {len(coeffs[-1])} coeffs")

# denoising
print(f"\n--- Wavelet Denoising ---")
for thresh in [0.1, 0.3, 0.5, 1.0]:
    denoised = denoise(noisy, thresh)
    mse = np.mean((clean - denoised)**2)
    print(f"  threshold={thresh}: MSE={mse:.4f}")

mse_noisy = np.mean((clean - noisy)**2)
print(f"  noisy signal MSE: {mse_noisy:.4f}")
