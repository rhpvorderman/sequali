import math
import sys

import dnaio

QUAL_TO_PHRED = [10 ** (-(i - 33) / 10)  for i in range(128)]


def fingerprint_sequence_original(sequence: str):
    if len(sequence) < 32:
        return sequence
    return sequence[:16] + sequence[-16:]


if __name__ == "__main__":
    expected_errors = [0 for _ in range(32 + 1)]
    fastq = sys.argv[1]
    with dnaio.open(fastq, mode="r", open_threads=1) as reader:
        for read in reader:  # type: dnaio.SequenceRecord
            fingerprint_quals = fingerprint_sequence_original(read.qualities)
            prob = 0.0
            for q in fingerprint_quals.encode("ascii"):
                prob += QUAL_TO_PHRED[q]
            expected_errors[round(prob)] += 1
    for i, count in enumerate(expected_errors):
        print(f"{i:2}\t{count:10,}")

