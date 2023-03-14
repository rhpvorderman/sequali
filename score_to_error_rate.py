if __name__ == "__main__":
    with open("src/fasterqc/score_to_error_rate.h", "wt") as output:
        output.write("// An array holding all error rates for each phred score.\n")
        output.write("// This file was automatically generated.\n\n")
        output.write("static const double SCORE_TO_ERROR_RATE[128] = {\n")
        for i in range(94):
            error_rate = 10 ** -(i/10)
            output.write(f"    {str(error_rate).upper() + 'L,':24}  // {i}\n")
        output.write("};\n")

