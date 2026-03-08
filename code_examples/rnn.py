"""RNN — vanilla recurrent neural network for sequence prediction."""

import numpy as np

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        s = 0.1
        self.Wxh = np.random.randn(input_dim, hidden_dim) * s
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * s
        self.Why = np.random.randn(hidden_dim, output_dim) * s
        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, xs):
        h = np.zeros(self.hidden_dim)
        hs, ys = [], []
        for x in xs:
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            y = h @ self.Why + self.by
            hs.append(h)
            ys.append(y)
        return ys, hs

    def predict_sequence(self, xs):
        ys, _ = self.forward(xs)
        return np.array(ys)

# --- demo: predict sin wave ---
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 100)
signal = np.sin(t)

# prepare sequences: predict next value from window of 5
window = 5
X, Y = [], []
for i in range(len(signal) - window):
    X.append(signal[i:i+window].reshape(-1, 1))
    Y.append(signal[i+window])
X, Y = np.array(X), np.array(Y)

rnn = SimpleRNN(1, 16, 1)

# simple training loop
lr = 0.001
for epoch in range(200):
    total_loss = 0
    for i in range(len(X)):
        ys, hs = rnn.forward(X[i])
        pred = ys[-1][0]
        loss = (pred - Y[i]) ** 2
        total_loss += loss
        # simple gradient for last output only
        dy = 2 * (pred - Y[i])
        rnn.Why -= lr * np.outer(hs[-1], [dy])
        rnn.by -= lr * dy
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}  MSE={total_loss/len(X):.4f}")

# test
preds = []
for i in range(len(X)):
    ys, _ = rnn.forward(X[i])
    preds.append(ys[-1][0])
preds = np.array(preds)
print(f"\nFinal MSE: {np.mean((preds - Y)**2):.4f}")
