"""FFT Applications — spectral analysis, filtering, and convolution."""

import numpy as np

def frequency_spectrum(signal, sample_rate):
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    magnitudes = np.abs(fft_vals) * 2 / n
    freqs = np.fft.rfftfreq(n, 1/sample_rate)
    return freqs, magnitudes

def lowpass_filter(signal, cutoff_freq, sample_rate):
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
    fft_vals[freqs > cutoff_freq] = 0
    return np.fft.irfft(fft_vals, len(signal))

def fft_convolve(a, b):
    n = len(a) + len(b) - 1
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    return np.fft.irfft(fa * fb, n)

# --- demo ---
np.random.seed(42)

# signal: 5 Hz + 20 Hz + noise
sample_rate = 100
t = np.arange(0, 2, 1/sample_rate)
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + np.random.randn(len(t)) * 0.3

freqs, mags = frequency_spectrum(signal, sample_rate)

print("=== FFT Applications ===\n")
print("--- Frequency Detection ---")
print(f"Signal: sin(2π·5t) + 0.5·sin(2π·20t) + noise")
# find peaks
peak_idx = np.where(mags > 0.3)[0]
for i in peak_idx:
    print(f"  Detected: {freqs[i]:.1f} Hz, magnitude={mags[i]:.3f}")

# lowpass filtering
print(f"\n--- Lowpass Filtering ---")
filtered = lowpass_filter(signal, cutoff_freq=10, sample_rate=sample_rate)
noise_before = np.std(signal - np.sin(2*np.pi*5*t))
noise_after = np.std(filtered - np.sin(2*np.pi*5*t))
print(f"  Cutoff: 10 Hz (removes 20 Hz component)")
print(f"  Noise std before: {noise_before:.3f}")
print(f"  Noise std after:  {noise_after:.3f}")

# FFT convolution speed
print(f"\n--- FFT Convolution ---")
a = np.random.randn(1000)
b = np.random.randn(100)
direct = np.convolve(a, b)
fft_result = fft_convolve(a, b)
print(f"  Direct convolution: O(n·m) = O({len(a)*len(b)})")
print(f"  FFT convolution:    O(n·log(n)) = O({int((len(a)+len(b))*np.log2(len(a)+len(b)))})")
print(f"  Max error: {np.max(np.abs(direct - fft_result)):.2e}")
