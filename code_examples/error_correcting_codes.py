"""Error Correcting Codes — Hamming(7,4) encode/decode."""

def hamming_encode(data_bits):
    """Encode 4 data bits into 7-bit Hamming code."""
    d1, d2, d3, d4 = data_bits
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]

def hamming_decode(code):
    """Decode 7-bit Hamming code, correcting single-bit errors."""
    s1 = code[0] ^ code[2] ^ code[4] ^ code[6]
    s2 = code[1] ^ code[2] ^ code[5] ^ code[6]
    s3 = code[3] ^ code[4] ^ code[5] ^ code[6]
    error_pos = s1 + s2 * 2 + s3 * 4
    if error_pos:
        code = list(code)
        code[error_pos - 1] ^= 1
        print(f"  Corrected bit at position {error_pos}")
    return [code[2], code[4], code[5], code[6]]

def flip_bit(code, pos):
    code = list(code)
    code[pos] ^= 1
    return code

# --- demo ---
data = [1, 0, 1, 1]
print("Original data:", data)

encoded = hamming_encode(data)
print("Encoded (7 bits):", encoded)

decoded = hamming_decode(encoded)
print("Decoded (no error):", decoded)

corrupted = flip_bit(encoded, 4)
print(f"\nCorrupted bit 4: {corrupted}")
recovered = hamming_decode(corrupted)
print("Recovered:", recovered)
print("Match:", recovered == data)
