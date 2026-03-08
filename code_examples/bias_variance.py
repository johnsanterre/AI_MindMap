"""Bias-Variance Tradeoff — polynomial regression at different complexities."""

import numpy as np

def poly_features(x, degree):
    return np.column_stack([x**i for i in range(degree+1)])

def fit_poly(x_train, y_train, degree):
    X = poly_features(x_train, degree)
    w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w

def predict_poly(x, w):
    X = poly_features(x, len(w)-1)
    return X @ w

def bias_variance_experiment(x_train_sets, y_train_sets, x_test, y_test, degree):
    predictions = []
    for xt, yt in zip(x_train_sets, y_train_sets):
        w = fit_poly(xt, yt, degree)
        predictions.append(predict_poly(x_test, w))
    preds = np.array(predictions)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(preds.var(axis=0))
    mse = np.mean((preds - y_test) ** 2)
    return bias2, variance, mse

# --- demo ---
np.random.seed(42)
true_fn = lambda x: np.sin(2 * x)

# generate multiple training sets
n_sets = 50
x_test = np.linspace(-3, 3, 100)
y_test = true_fn(x_test)

train_sets_x, train_sets_y = [], []
for _ in range(n_sets):
    x = np.random.uniform(-3, 3, 20)
    y = true_fn(x) + np.random.randn(20) * 0.3
    train_sets_x.append(x)
    train_sets_y.append(y)

print(f"{'Degree':>7} {'Bias²':>8} {'Variance':>10} {'MSE':>8}")
print("-" * 37)
for deg in [1, 2, 3, 5, 10, 15]:
    b2, var, mse = bias_variance_experiment(train_sets_x, train_sets_y, x_test, y_test, deg)
    print(f"{deg:>7} {b2:>8.4f} {var:>10.4f} {mse:>8.4f}")
