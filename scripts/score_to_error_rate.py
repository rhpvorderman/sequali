if __name__ == "__main__":
    max_phred = 93
    with open("src/sequali/score_to_error_rate.h", "wt") as output:
        output.write("// An array holding all error rates for each phred score.\n")
        output.write("// This file was automatically generated.\n\n")
        output.write(f"static const double SCORE_TO_ERROR_RATE[{max_phred + 1}] = " + "{\n")
        for i in range(max_phred + 1):
            error_rate = 10 ** -(i/10)
            output.write(f"    {str(error_rate).upper() + 'L,':24}  // {i}\n")
        output.write("};\n")

