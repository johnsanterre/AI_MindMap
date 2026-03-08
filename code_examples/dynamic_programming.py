"""Dynamic Programming — classic DP problems."""

import numpy as np

def knapsack_01(weights, values, capacity):
    """0/1 Knapsack problem."""
    n = len(weights)
    dp = np.zeros((n+1, capacity+1))
    for i in range(1, n+1):
        for w in range(capacity+1):
            dp[i, w] = dp[i-1, w]
            if weights[i-1] <= w:
                dp[i, w] = max(dp[i, w], dp[i-1, w-weights[i-1]] + values[i-1])
    # backtrack
    items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i-1, w]:
            items.append(i-1)
            w -= weights[i-1]
    return int(dp[n, capacity]), items[::-1]

def longest_common_subseq(s1, s2):
    """LCS with backtracking."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1] + 1
            else:
                dp[i, j] = max(dp[i-1, j], dp[i, j-1])
    # backtrack
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1; j -= 1
        elif dp[i-1, j] > dp[i, j-1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(lcs))

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1), dtype=int)
    dp[:, 0] = np.arange(m+1)
    dp[0, :] = np.arange(n+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return int(dp[m, n])

# --- demo ---
print("=== 0/1 Knapsack ===")
weights = [2, 3, 4, 5]
values =  [3, 4, 5, 6]
max_val, items = knapsack_01(weights, values, capacity=8)
print(f"  Items: {items}, Value: {max_val}, Weight: {sum(weights[i] for i in items)}")

print("\n=== Longest Common Subsequence ===")
s1, s2 = "ABCBDAB", "BDCAB"
lcs = longest_common_subseq(s1, s2)
print(f"  LCS(\"{s1}\", \"{s2}\") = \"{lcs}\" (length {len(lcs)})")

print("\n=== Edit Distance ===")
pairs = [("kitten", "sitting"), ("sunday", "saturday"), ("hello", "hello")]
for a, b in pairs:
    print(f"  edit(\"{a}\", \"{b}\") = {edit_distance(a, b)}")
