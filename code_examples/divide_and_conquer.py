"""Divide and Conquer — binary search, Karatsuba multiplication, closest pair."""

def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    p = 10 ** m
    a, b = divmod(x, p)
    c, d = divmod(y, p)
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    abcd = karatsuba(a + b, c + d) - ac - bd
    return ac * (10 ** (2 * m)) + abcd * (10 ** m) + bd

def closest_pair_1d(points):
    """Closest pair in 1D (simplified to show the D&C structure)."""
    pts = sorted(points)
    if len(pts) <= 3:
        best = float('inf')
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                best = min(best, abs(pts[i] - pts[j]))
        return best
    mid = len(pts) // 2
    d = min(closest_pair_1d(pts[:mid]), closest_pair_1d(pts[mid:]))
    # check across the split
    d = min(d, abs(pts[mid] - pts[mid-1]))
    return d

# --- demo ---
print("Binary search for 7:", binary_search([1,3,5,7,9,11], 7))
print("Karatsuba 1234 × 5678 =", karatsuba(1234, 5678))
print("Verify:                ", 1234 * 5678)
print("Closest pair:", closest_pair_1d([2, 15, 4, 12, 7, 10, 3]))
