"""Huffman Coding — optimal prefix-free compression."""

import numpy as np
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

def build_huffman(text):
    freqs = Counter(text)
    nodes = [HuffmanNode(ch, f) for ch, f in freqs.items()]

    while len(nodes) > 1:
        nodes.sort(key=lambda n: n.freq)
        left, right = nodes.pop(0), nodes.pop(0)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        nodes.append(parent)

    codes = {}
    def traverse(node, code=""):
        if node.char is not None:
            codes[node.char] = code if code else "0"
            return
        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(nodes[0])
    return codes, freqs

def encode(text, codes):
    return "".join(codes[ch] for ch in text)

def entropy(freqs):
    total = sum(freqs.values())
    return -sum(f/total * np.log2(f/total) for f in freqs.values())

# --- demo ---
text = "abracadabra alakazam"

codes, freqs = build_huffman(text)
encoded = encode(text, codes)

print("=== Huffman Coding ===\n")
print(f"Text: '{text}' ({len(text)} chars)\n")

print(f"{'Char':>6} {'Freq':>5} {'Code':>8} {'Bits':>5}")
print("-" * 28)
for ch in sorted(codes, key=lambda c: freqs[c], reverse=True):
    print(f"{'(sp)' if ch == ' ' else ch:>6} {freqs[ch]:>5} {codes[ch]:>8} {len(codes[ch]):>5}")

orig_bits = len(text) * 8  # ASCII
huff_bits = len(encoded)
H = entropy(freqs)
avg_bits = sum(len(codes[ch]) * freqs[ch] for ch in codes) / len(text)

print(f"\nOriginal:  {orig_bits} bits (8 bits/char)")
print(f"Huffman:   {huff_bits} bits ({avg_bits:.2f} bits/char)")
print(f"Entropy:   {H:.2f} bits/char (theoretical minimum)")
print(f"Ratio:     {huff_bits/orig_bits:.1%}")
print(f"Huffman is within {avg_bits - H:.3f} bits/char of optimal.")
