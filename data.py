import csv

def pcg16(seed: int, sequence: int, count: int):
    """Generate 16-bit PCG (XSH-RR) pseudo-random numbers as bipolar bit vectors.

    Yields `count` lists of 16 values in {-1, 1}, one per generated number.
    """
    multiplier = 747796405
    inc = (sequence << 1) | 1

    # PCG state initialization
    state = (0 + inc) & 0xFFFFFFFF
    state = (state + seed) & 0xFFFFFFFF
    state = (state * multiplier + inc) & 0xFFFFFFFF

    for _ in range(count):
        old_state = state
        state = (old_state * multiplier + inc) & 0xFFFFFFFF

        # XSH-RR: xorshift high, random rotate
        xorshifted = (((old_state >> 16) ^ old_state) >> 5) & 0xFFFF
        rot = old_state >> 27
        value = ((xorshifted >> rot) | (xorshifted << ((-rot) & 15))) & 0xFFFF
        # Map binary digits to {-1, 1}: 0 -> -1, 1 -> 1
        bits = [int(digit) * 2 - 1 for digit in f"{value:016b}"]
        yield bits
    

if __name__ == "__main__":
    numbers = pcg16(seed=1, sequence=0, count=65536)

    with open("data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        for bits in numbers:
            writer.writerow(bits)
