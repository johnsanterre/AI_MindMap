"""RSA Cryptography — toy RSA implementation."""

import random

def is_prime(n, k=10):
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

def gen_prime(bits=16):
    while True:
        p = random.getrandbits(bits) | (1 << bits - 1) | 1
        if is_prime(p):
            return p

def extended_gcd(a, b):
    if b == 0: return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def rsa_keygen(bits=16):
    p, q = gen_prime(bits), gen_prime(bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    _, d, _ = extended_gcd(e, phi)
    d %= phi
    return (e, n), (d, n)

def encrypt(msg, pub):
    e, n = pub
    return pow(msg, e, n)

def decrypt(ct, priv):
    d, n = priv
    return pow(ct, d, n)

# --- demo ---
pub, priv = rsa_keygen(32)
print(f"Public key:  e={pub[0]}, n={pub[1]}")
print(f"Private key: d={priv[0]}")

message = 42
ct = encrypt(message, pub)
pt = decrypt(ct, priv)
print(f"\nMessage:    {message}")
print(f"Encrypted:  {ct}")
print(f"Decrypted:  {pt}")
print(f"Round-trip:  {'✓' if pt == message else '✗'}")
