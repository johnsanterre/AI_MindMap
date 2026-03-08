"""Combinatorics — permutations, combinations, generating subsets."""

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def P(n, k):
    """Permutations: n! / (n-k)!"""
    return factorial(n) // factorial(n - k)

def C(n, k):
    """Combinations: n! / (k! * (n-k)!)"""
    return factorial(n) // (factorial(k) * factorial(n - k))

def permutations(items):
    if len(items) <= 1:
        return [items]
    result = []
    for i, item in enumerate(items):
        rest = items[:i] + items[i+1:]
        for perm in permutations(rest):
            result.append([item] + perm)
    return result

def combinations(items, k):
    if k == 0:
        return [[]]
    if not items:
        return []
    first, rest = items[0], items[1:]
    with_first = [[first] + c for c in combinations(rest, k-1)]
    without = combinations(rest, k)
    return with_first + without

def power_set(items):
    if not items:
        return [[]]
    rest = power_set(items[1:])
    return rest + [[items[0]] + s for s in rest]

# --- demo ---
print(f"P(5,3) = {P(5,3)}")
print(f"C(5,3) = {C(5,3)}")
print(f"\nPermutations of [1,2,3]: {permutations([1,2,3])}")
print(f"\nC(4,2) combinations of [A,B,C,D]: {combinations(['A','B','C','D'], 2)}")
print(f"\nPower set of [1,2,3]: {power_set([1,2,3])}")

# Pascal's triangle
print("\nPascal's triangle (first 8 rows):")
for n in range(8):
    row = [C(n, k) for k in range(n+1)]
    print(f"  {'  ' * (7-n)}{' '.join(f'{x:3}' for x in row)}")
