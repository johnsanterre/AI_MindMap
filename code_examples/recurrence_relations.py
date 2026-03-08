"""Recurrence Relations — Fibonacci, Tower of Hanoi, memoization."""

def fib_naive(n):
    """Exponential: T(n) = T(n-1) + T(n-2)."""
    if n <= 1: return n
    return fib_naive(n-1) + fib_naive(n-2)

def fib_memo(n, memo={}):
    """O(n) with memoization."""
    if n <= 1: return n
    if n not in memo:
        memo[n] = fib_memo(n-1) + fib_memo(n-2)
    return memo[n]

def fib_matrix(n):
    """O(log n) via matrix exponentiation."""
    def mat_mul(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
        ]
    def mat_pow(M, p):
        if p == 1: return M
        if p % 2 == 0:
            half = mat_pow(M, p // 2)
            return mat_mul(half, half)
        return mat_mul(M, mat_pow(M, p - 1))
    if n <= 1: return n
    result = mat_pow([[1,1],[1,0]], n)
    return result[0][1]

def hanoi(n, src="A", dst="C", aux="B"):
    """T(n) = 2T(n-1) + 1 → 2^n - 1 moves."""
    if n == 1:
        return [(src, dst)]
    moves = hanoi(n-1, src, aux, dst)
    moves.append((src, dst))
    moves += hanoi(n-1, aux, dst, src)
    return moves

# --- demo ---
print("Fibonacci (first 15):")
print([fib_memo(i) for i in range(15)])
print(f"fib(50) = {fib_memo(50)}")
print(f"fib(50) via matrix = {fib_matrix(50)}")

print(f"\nTower of Hanoi (3 disks): {len(hanoi(3))} moves")
for src, dst in hanoi(3):
    print(f"  {src} → {dst}")
