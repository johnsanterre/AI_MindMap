"""SMOTE — Synthetic Minority Over-sampling Technique."""

import numpy as np

def smote(X_minority, n_synthetic, k=5):
    """Generate synthetic samples by interpolating between neighbors."""
    n = len(X_minority)
    # find k nearest neighbors for each minority sample
    dists = np.sum((X_minority[:, None] - X_minority[None, :])**2, axis=2)
    np.fill_diagonal(dists, np.inf)
    knn_indices = np.argsort(dists, axis=1)[:, :k]

    synthetic = []
    for _ in range(n_synthetic):
        i = np.random.randint(n)
        nn = knn_indices[i, np.random.randint(k)]
        lam = np.random.rand()
        new_sample = X_minority[i] + lam * (X_minority[nn] - X_minority[i])
        synthetic.append(new_sample)

    return np.array(synthetic)

def random_oversample(X_minority, n_synthetic):
    idx = np.random.choice(len(X_minority), n_synthetic, replace=True)
    return X_minority[idx]

# --- demo ---
np.random.seed(42)

# imbalanced dataset
n_majority = 200
n_minority = 20
d = 4

X_maj = np.random.randn(n_majority, d) + [2, 0, 0, 0]
X_min = np.random.randn(n_minority, d) + [-2, 0, 0, 0]
X = np.vstack([X_maj, X_min])
y = np.array([0]*n_majority + [1]*n_minority)

print("=== SMOTE ===")
print(f"Original: {n_majority} majority, {n_minority} minority (ratio {n_majority/n_minority:.0f}:1)\n")

# generate synthetic minority samples
n_synthetic = n_majority - n_minority  # balance the classes
X_synthetic = smote(X_min, n_synthetic, k=5)
X_random = random_oversample(X_min, n_synthetic)

print(f"Synthetic samples generated: {n_synthetic}")
print(f"New balance: {n_majority} : {n_minority + n_synthetic} (1:1)\n")

# compare distributions
print("--- Distribution Check ---")
print(f"{'':>20} {'Mean':>8} {'Std':>8}")
print(f"{'Original minority':>20} {X_min[:, 0].mean():>8.3f} {X_min[:, 0].std():>8.3f}")
print(f"{'SMOTE synthetic':>20} {X_synthetic[:, 0].mean():>8.3f} {X_synthetic[:, 0].std():>8.3f}")
print(f"{'Random oversample':>20} {X_random[:, 0].mean():>8.3f} {X_random[:, 0].std():>8.3f}")

# verify SMOTE creates diverse samples
unique_smote = len(np.unique(X_synthetic, axis=0))
unique_random = len(np.unique(X_random, axis=0))
print(f"\nUnique SMOTE samples:  {unique_smote}/{n_synthetic}")
print(f"Unique random samples: {unique_random}/{n_synthetic}")
print(f"\nSMOTE creates novel in-between points. Random just duplicates.")
