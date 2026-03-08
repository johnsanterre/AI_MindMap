"""Stratified Sampling — ensuring representative subgroups in samples."""

import numpy as np

def random_split(X, y, test_ratio=0.2):
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def stratified_split(X, y, test_ratio=0.2):
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        split = int(len(cls_idx) * (1 - test_ratio))
        train_idx.extend(cls_idx[:split])
        test_idx.extend(cls_idx[split:])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def stratified_kfold(X, y, k=5):
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % k].append(idx)
    return [np.array(f) for f in folds]

# --- demo ---
np.random.seed(42)

# imbalanced dataset: 80% class 0, 15% class 1, 5% class 2
n = 200
y = np.array([0]*160 + [1]*30 + [2]*10)
X = np.random.randn(n, 3)

print("=== Stratified Sampling ===")
print(f"Full dataset class distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

# compare random vs stratified (run 10 trials)
print(f"{'':>12} {'Class 0 %':>10} {'Class 1 %':>10} {'Class 2 %':>10}")
print("-" * 46)

# full data proportions
props = np.array([160, 30, 10]) / 200
print(f"{'Full data':>12} {props[0]:>10.1%} {props[1]:>10.1%} {props[2]:>10.1%}")

random_props = []
strat_props = []
for _ in range(20):
    _, X_te_r, _, y_te_r = random_split(X, y)
    _, X_te_s, _, y_te_s = stratified_split(X, y)
    r_prop = [np.mean(y_te_r == c) for c in range(3)]
    s_prop = [np.mean(y_te_s == c) for c in range(3)]
    random_props.append(r_prop)
    strat_props.append(s_prop)

random_props = np.array(random_props)
strat_props = np.array(strat_props)

print(f"{'Random avg':>12} {random_props[:,0].mean():>10.1%} {random_props[:,1].mean():>10.1%} {random_props[:,2].mean():>10.1%}")
print(f"{'Random std':>12} {random_props[:,0].std():>10.1%} {random_props[:,1].std():>10.1%} {random_props[:,2].std():>10.1%}")
print(f"{'Strat avg':>12} {strat_props[:,0].mean():>10.1%} {strat_props[:,1].mean():>10.1%} {strat_props[:,2].mean():>10.1%}")
print(f"{'Strat std':>12} {strat_props[:,0].std():>10.1%} {strat_props[:,1].std():>10.1%} {strat_props[:,2].std():>10.1%}")

print(f"\nStratified sampling preserves class ratios with much less variance.")
