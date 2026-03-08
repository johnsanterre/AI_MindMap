"""LSTM — Long Short-Term Memory with gating mechanism."""

import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
def tanh(x): return np.tanh(x)

class LSTM:
    def __init__(self, input_size, hidden_size):
        s = np.sqrt(2 / (input_size + hidden_size))
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * s
        self.bf = np.ones(hidden_size)   # forget bias initialized to 1
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def step(self, x, h_prev, c_prev):
        concat = np.concatenate([h_prev, x])
        f = sigmoid(concat @ self.Wf + self.bf)  # forget gate
        i = sigmoid(concat @ self.Wi + self.bi)  # input gate
        c_hat = tanh(concat @ self.Wc + self.bc)  # candidate
        o = sigmoid(concat @ self.Wo + self.bo)  # output gate

        c = f * c_prev + i * c_hat  # cell state
        h = o * tanh(c)             # hidden state
        return h, c, {'f': f, 'i': i, 'o': o}

    def forward(self, X):
        """X: (seq_len, input_size)."""
        T = len(X)
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        hiddens, gates_list = [], []
        for t in range(T):
            h, c, gates = self.step(X[t], h, c)
            hiddens.append(h)
            gates_list.append(gates)
        return np.array(hiddens), gates_list

# --- demo ---
np.random.seed(42)

lstm = LSTM(input_size=3, hidden_size=4)

# sequence with a signal that must be remembered
T = 10
X = np.random.randn(T, 3) * 0.1
X[2] = [5, 0, 0]  # important signal at t=2

hiddens, gates_list = lstm.forward(X)

print("=== LSTM: Long Short-Term Memory ===")
print(f"Input: ({T}, 3), Hidden: 4\n")

print(f"{'t':>3} {'||h||':>8} {'Forget':>8} {'Input':>8} {'Output':>8}")
print("-" * 38)
for t in range(T):
    f_avg = gates_list[t]['f'].mean()
    i_avg = gates_list[t]['i'].mean()
    o_avg = gates_list[t]['o'].mean()
    h_norm = np.linalg.norm(hiddens[t])
    marker = " ← signal" if t == 2 else ""
    print(f"{t:>3} {h_norm:>8.4f} {f_avg:>8.3f} {i_avg:>8.3f} {o_avg:>8.3f}{marker}")

print(f"\nForget gate ≈ 1: keep memory. Input gate ≈ 1: write new info.")
print(f"This gating prevents vanishing gradients over long sequences.")
