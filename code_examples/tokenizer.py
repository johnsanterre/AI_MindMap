"""Tokenizer — Byte Pair Encoding (BPE) from scratch."""

from collections import Counter

def get_pairs(word_freqs):
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    new_freqs = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in word_freqs.items():
        new_word = word.replace(bigram, replacement)
        new_freqs[new_word] = freq
    return new_freqs

def learn_bpe(corpus, num_merges=20):
    # initialize: split words into characters
    words = corpus.split()
    word_counts = Counter(words)
    word_freqs = {}
    for word, count in word_counts.items():
        word_freqs[' '.join(list(word)) + ' </w>'] = count

    merges = []
    for i in range(num_merges):
        pairs = get_pairs(word_freqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        word_freqs = merge_pair(best, word_freqs)
        if i < 10 or i == num_merges - 1:
            print(f"  Merge {i+1:2d}: {best[0]:>6} + {best[1]:<6} (freq={pairs[best]})")

    return merges, word_freqs

def tokenize(word, merges):
    tokens = list(word) + ['</w>']
    for a, b in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i+1] == b:
                tokens = tokens[:i] + [a+b] + tokens[i+2:]
            else:
                i += 1
    return tokens

# --- demo ---
corpus = ("the cat sat on the mat . the cat ate the rat . "
          "the dog sat on the log . the dog ate the frog .")

print("Learning BPE merges:")
merges, vocab = learn_bpe(corpus, num_merges=20)

print(f"\nFinal vocabulary ({len(vocab)} entries):")
for word, freq in sorted(vocab.items(), key=lambda x: -x[1])[:10]:
    print(f"  {word:<20} freq={freq}")

print("\nTokenization examples:")
for word in ["cat", "sat", "frog", "rating"]:
    tokens = tokenize(word, merges)
    print(f"  {word:>8} → {tokens}")
