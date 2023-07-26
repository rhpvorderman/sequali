if __name__ == "__main__":
    TWOBIT_TO_NUCLEOTIDE = ('A', 'C', 'G', 'T')
    with open("src/sequali/twobit_to_nucleotides.h", "wt") as output:
        output.write("// An array holding all nucleotide 4 pairs per two-bit uint8_t.\n")
        output.write("// This file was automatically generated.\n\n")
        output.write(f"static const uint8_t TWOBIT_TO_NUCLEOTIDES[4][256] = " + "{\n")
        for i in range(2**8):
            one = TWOBIT_TO_NUCLEOTIDE[(i >> 6) & 0b11]
            two = TWOBIT_TO_NUCLEOTIDE[(i >> 4) & 0b11] 
            three = TWOBIT_TO_NUCLEOTIDE[(i >> 2) & 0b11]
            four = TWOBIT_TO_NUCLEOTIDE[i &0b11]

            
            output.write(f"    {{'{one}', '{two}', '{three}', '{four}'}},  // {i}\n")
        output.write("};\n")

