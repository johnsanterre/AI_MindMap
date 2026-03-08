"""Modular Arithmetic — GCD, modular inverse, fast exponentiation."""

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Returns (g, x, y) such that a*x + b*y = g = gcd(a,b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def mod_inverse(a, m):
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None  # no inverse
    return x % m

def mod_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        exp >>= 1
        base = base * base % mod
    return result

# --- demo ---
print("gcd(48, 18) =", gcd(48, 18))
print("7^-1 mod 11 =", mod_inverse(7, 11))
print("Verify: 7 *", mod_inverse(7, 11), "mod 11 =", (7 * mod_inverse(7, 11)) % 11)
print("2^100 mod 997 =", mod_pow(2, 100, 997))
print("Built-in check:", pow(2, 100, 997))
