import csv

def pcg16(seed: int, sequence : int, count: int):
    # Standard 32-bit LCG multiplier from PCG reference
    # Note: 747796405 is also common, but 22695477 is the library default
    multiplier = 747796405 
    inc = (sequence << 1) | 1
    
    # --- Official Initialization ---
    state = (0 + inc) & 0xFFFFFFFF
    state = (state + seed) & 0xFFFFFFFF
    state = (state * multiplier + inc) & 0xFFFFFFFF
    
    for _ in range(count):
        old_state = state
        # Step the LCG
        state = (old_state * multiplier + inc) & 0xFFFFFFFF
        
        # XSH-RR (Xorshift High, Random Rotate)
        # Shift 16 bits down, XOR with self, then shift 5 for the 16-bit result
        xorshifted = (((old_state >> 16) ^ old_state) >> 5) & 0xFFFF
        
        # Determine rotation amount from the top 5 bits
        rot = old_state >> 27
        
        # 16-bit circular rotation (standardized)
        value = ((xorshifted >> rot) | (xorshifted << ((-rot) & 15))) & 0xFFFF
        # prolly a better way but idc
        num = [int(digit) - 1 if int(digit) == 0 else int(digit) for digit in list(f"{value:016b}") ]  
        yield num        
    

numbers = pcg16(seed=1, sequence=0, count=65536)

with open('data.csv', mode='a', newline='') as file:
    for num in numbers:
        writer = csv.writer(file)
        writer.writerow(num)

    
