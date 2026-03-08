"""Primality Testing — trial division, Fermat, Miller-Rabin."""

import random

def trial_division(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def fermat_test(n, k=5):
    """Probabilistic — fooled by Carmichael numbers."""
    if n < 2: return False
    if n < 4: return True
    for _ in range(k):
        a = random.randrange(2, n)
        if pow(a, n - 1, n) != 1:
            return False
    return True

def miller_rabin(n, k=10):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# --- demo ---
test_nums = [2, 17, 561, 997, 1009, 1729, 7919, 104729]
print(f"{'n':>8}  {'Trial':>7}  {'Fermat':>7}  {'M-R':>7}")
print("-" * 38)
for n in test_nums:
    t = trial_division(n)
    f = fermat_test(n)
    m = miller_rabin(n)
    flag = " ← Carmichael!" if f and not t else ""
    print(f"{n:>8}  {str(t):>7}  {str(f):>7}  {str(m):>7}{flag}")
