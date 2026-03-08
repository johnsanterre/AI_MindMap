"""Mixture of Experts — routing inputs to specialized sub-networks."""

import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def relu(x): return np.maximum(0, x)

class Expert:
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2/d_in)
        self.b = np.zeros(d_out)

    def forward(self, x):
        return relu(x @ self.W + self.b)

class MixtureOfExperts:
    def __init__(self, d_in, d_hidden, d_out, n_experts, top_k=2):
        self.experts = [Expert(d_in, d_hidden) for _ in range(n_experts)]
        self.gate_W = np.random.randn(d_in, n_experts) * 0.1
        self.output_W = np.random.randn(d_hidden, d_out) * np.sqrt(2/d_hidden)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, X):
        # gating: which experts to use
        gate_logits = X @ self.gate_W
        gate_probs = softmax(gate_logits)

        # top-k routing
        top_k_idx = np.argsort(gate_probs, axis=1)[:, -self.top_k:]
        outputs = np.zeros((len(X), self.output_W.shape[1]))

        expert_counts = np.zeros(self.n_experts)

        for i in range(len(X)):
            expert_out = np.zeros(self.experts[0].W.shape[1])
            weight_sum = 0
            for k in top_k_idx[i]:
                w = gate_probs[i, k]
                expert_out += w * self.experts[k].forward(X[i:i+1])[0]
                weight_sum += w
                expert_counts[k] += 1
            expert_out /= (weight_sum + 1e-8)
            outputs[i] = expert_out @ self.output_W

        return outputs, gate_probs, expert_counts

# --- demo ---
np.random.seed(42)

n_experts = 4
moe = MixtureOfExperts(d_in=8, d_hidden=16, d_out=3, n_experts=n_experts, top_k=2)

# different input types (simulating specialization)
X1 = np.random.randn(20, 8) + [2, 0, 0, 0, 0, 0, 0, 0]
X2 = np.random.randn(20, 8) + [0, 0, 2, 0, 0, 0, 0, 0]
X3 = np.random.randn(20, 8) + [0, 0, 0, 0, 2, 0, 0, 0]
X = np.vstack([X1, X2, X3])

outputs, gate_probs, counts = moe.forward(X)

print("=== Mixture of Experts ===")
print(f"Experts: {n_experts}, Top-k: 2, Input dim: 8\n")

print("Expert utilization:")
for e in range(n_experts):
    print(f"  Expert {e}: {int(counts[e])} tokens ({counts[e]/len(X):.0%})")

print(f"\nAvg routing weights per input group:")
print(f"{'Group':>8}", end="")
for e in range(n_experts):
    print(f"  E{e:>5}", end="")
print()
for name, start, end in [("Type A", 0, 20), ("Type B", 20, 40), ("Type C", 40, 60)]:
    avg_gate = gate_probs[start:end].mean(axis=0)
    print(f"{name:>8}", end="")
    for e in range(n_experts):
        print(f"  {avg_gate[e]:>5.3f}", end="")
    print()

print(f"\nTotal params: {sum(e.W.size + e.b.size for e in moe.experts) + moe.gate_W.size + moe.output_W.size}")
print(f"Active params per token: ~{moe.experts[0].W.size * 2 + moe.output_W.size} (top-2)")
