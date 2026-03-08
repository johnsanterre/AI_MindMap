"""Isotonic Regression — monotonic fitting via pool adjacent violators."""

import numpy as np

def isotonic_regression(y, increasing=True):
    """Pool Adjacent Violators Algorithm (PAVA)."""
    n = len(y)
    result = y.copy().astype(float)
    if not increasing:
        result = -result

    # merge blocks that violate monotonicity
    blocks = [[i] for i in range(n)]
    values = list(result)

    changed = True
    while changed:
        changed = False
        i = 0
        new_blocks = []
        new_values = []
        while i < len(blocks):
            if i + 1 < len(blocks) and values[i] > values[i+1]:
                # merge blocks
                merged = blocks[i] + blocks[i+1]
                merged_val = np.mean(result[merged])
                new_blocks.append(merged)
                new_values.append(merged_val)
                changed = True
                i += 2
            else:
                new_blocks.append(blocks[i])
                new_values.append(values[i])
                i += 1
        blocks = new_blocks
        values = new_values

    output = np.zeros(n)
    for block, val in zip(blocks, values):
        for idx in block:
            output[idx] = val

    if not increasing:
        output = -output
    return output

# --- demo ---
np.random.seed(42)

# noisy monotonic data
n = 20
x = np.sort(np.random.uniform(0, 10, n))
y_true = 2 * np.sqrt(x)
y_noisy = y_true + np.random.randn(n) * 0.8

y_iso = isotonic_regression(y_noisy)

print("=== Isotonic Regression ===")
print("Fits a monotonically increasing function\n")

print(f"{'x':>6} {'y_true':>8} {'y_noisy':>9} {'y_iso':>8}")
print("-" * 34)
for i in range(n):
    print(f"{x[i]:>6.2f} {y_true[i]:>8.3f} {y_noisy[i]:>9.3f} {y_iso[i]:>8.3f}")

mse_noisy = np.mean((y_true - y_noisy)**2)
mse_iso = np.mean((y_true - y_iso)**2)
print(f"\nMSE (noisy):    {mse_noisy:.4f}")
print(f"MSE (isotonic): {mse_iso:.4f}")

# verify monotonicity
diffs = np.diff(y_iso)
print(f"Monotonic: {np.all(diffs >= -1e-10)}")
print(f"\nUsed in: probability calibration, dose-response curves.")
