"""Beam Search — sequence decoding with beam width."""

import numpy as np

def greedy_decode(log_probs, max_len=8):
    """Greedy: always pick highest probability token."""
    seq = [0]  # start token
    for t in range(max_len):
        probs = log_probs[seq[-1]]
        next_tok = probs.argmax()
        if next_tok == 1:  # end token
            break
        seq.append(next_tok)
    return seq, sum(log_probs[seq[i]][seq[i+1]] for i in range(len(seq)-1))

def beam_search(log_probs, beam_width=3, max_len=8):
    """Beam search: keep top-k candidates at each step."""
    vocab_size = log_probs.shape[1]
    beams = [(0.0, [0])]  # (score, sequence)

    for t in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == 1:  # end token, keep as-is
                candidates.append((score, seq))
                continue
            probs = log_probs[seq[-1]]
            for tok in range(vocab_size):
                candidates.append((score + probs[tok], seq + [tok]))
        candidates.sort(key=lambda x: -x[0])
        beams = candidates[:beam_width]
        if all(seq[-1] == 1 for _, seq in beams):
            break

    return beams

# --- demo ---
np.random.seed(42)
vocab = ["<s>", "</s>", "the", "cat", "dog", "sat", "ran", "fast", "slow"]
V = len(vocab)

# random log-probability transitions (rows = current token, cols = next token)
raw = np.random.randn(V, V)
raw[:, 0] = -10  # never go back to start
log_probs = raw - np.log(np.exp(raw).sum(axis=1, keepdims=True))

print("=== Greedy Decode ===")
seq, score = greedy_decode(log_probs)
print(f"  {' '.join(vocab[t] for t in seq)} (score={score:.3f})")

print(f"\n=== Beam Search ===")
for width in [1, 3, 5]:
    results = beam_search(log_probs, beam_width=width)
    best_score, best_seq = results[0]
    print(f"  width={width}: {' '.join(vocab[t] for t in best_seq)} (score={best_score:.3f})")
