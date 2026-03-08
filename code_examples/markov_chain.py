"""Markov Chain — transition matrix, stationary distribution, text generation."""

import numpy as np
from collections import defaultdict

def stationary_distribution(P, tol=1e-8, max_iter=1000):
    """Find stationary distribution π such that πP = π."""
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = pi @ P
        if np.abs(pi_new - pi).max() < tol:
            break
        pi = pi_new
    return pi

def simulate_chain(P, states, start=0, steps=20):
    path = [start]
    current = start
    for _ in range(steps):
        current = np.random.choice(len(states), p=P[current])
        path.append(current)
    return [states[i] for i in path]

def text_markov(corpus, order=1):
    """Build a character-level Markov chain from text."""
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(corpus) - order):
        state = corpus[i:i+order]
        next_char = corpus[i+order]
        transitions[state][next_char] += 1
    return transitions

def generate_text(transitions, start, length=100):
    result = start
    current = start
    for _ in range(length):
        if current not in transitions:
            break
        chars = list(transitions[current].keys())
        counts = list(transitions[current].values())
        probs = np.array(counts) / sum(counts)
        next_char = np.random.choice(chars, p=probs)
        result += next_char
        current = current[1:] + next_char
    return result

# --- demo ---
np.random.seed(42)

# 1) Weather Markov chain
states = ["Sunny", "Cloudy", "Rainy"]
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])

pi = stationary_distribution(P)
print("Weather Markov Chain:")
print(f"  Stationary dist: {dict(zip(states, pi.round(3)))}")
print(f"  Sample path: {' → '.join(simulate_chain(P, states, steps=10))}")

# 2) Text generation
print("\nText Markov Chain (order=2):")
corpus = "the quick brown fox jumps over the lazy dog and the quick red fox runs fast"
trans = text_markov(corpus, order=2)
for _ in range(3):
    print(f"  \"{generate_text(trans, 'th', length=40)}\"")
