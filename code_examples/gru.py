"""GRU — Gated Recurrent Unit, a simpler alternative to LSTM."""

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

class GRU:
    def __init__(self, input_size, hidden_size):
        s = np.sqrt(2 / (input_size + hidden_size))
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def step(self, x, h_prev):
        concat = np.concatenate([h_prev, x])
        z = sigmoid(concat @ self.Wz + self.bz)  # update gate
        r = sigmoid(concat @ self.Wr + self.br)  # reset gate
        concat_r = np.concatenate([r * h_prev, x])
        h_hat = np.tanh(concat_r @ self.Wh + self.bh)  # candidate
        h = (1 - z) * h_prev + z * h_hat  # interpolate
        return h, {'z': z, 'r': r}

    def forward(self, X):
        T = len(X)
        h = np.zeros(self.hidden_size)
        hiddens, gates_list = [], []
        for t in range(T):
            h, gates = self.step(X[t], h)
            hiddens.append(h)
            gates_list.append(gates)
        return np.array(hiddens), gates_list

# --- demo ---
np.random.seed(42)

print("=== GRU vs LSTM Comparison ===\n")

input_size, hidden_size = 3, 4
gru = GRU(input_size, hidden_size)

# parameter count comparison
gru_params = 3 * (input_size + hidden_size) * hidden_size + 3 * hidden_size
lstm_params = 4 * (input_size + hidden_size) * hidden_size + 4 * hidden_size
print(f"GRU parameters:  {gru_params}")
print(f"LSTM parameters: {lstm_params}")
print(f"GRU is {lstm_params/gru_params:.0%} the size of LSTM\n")

# sequence processing
T = 8
X = np.random.randn(T, input_size) * 0.5
X[1] = [3, 0, 0]  # important signal

hiddens, gates = gru.forward(X)

print(f"{'t':>3} {'||h||':>8} {'Update z':>10} {'Reset r':>10}")
print("-" * 34)
for t in range(T):
    z = gates[t]['z'].mean()
    r = gates[t]['r'].mean()
    h = np.linalg.norm(hiddens[t])
    marker = " ← signal" if t == 1 else ""
    print(f"{t:>3} {h:>8.4f} {z:>10.3f} {r:>10.3f}{marker}")

print(f"\nUpdate gate z: how much new info to incorporate")
print(f"Reset gate r: how much past to forget")
print(f"GRU merges LSTM's forget+input gates into one update gate.")
