import math
import sys

import dnaio

QUAL_TO_PHRED = [10 ** (-(i - 33) / 10)  for i in range(128)]


def fingerprint_sequence_original(sequence: str):
    if len(sequence) < 32:
        return sequence
    return sequence[:16] + sequence[-16:]


if __name__ == "__main__":
    phred_scores = [0 for _ in range(94)]
    fastq = sys.argv[1]
    with dnaio.open(fastq, mode="r") as reader:
        for read in reader:  # type: dnaio.SequenceRecord
            fingerprint_quals = fingerprint_sequence_original(read.qualities)
            prob = 0.0
            for q in fingerprint_quals.encode("ascii"):
                prob += QUAL_TO_PHRED[q]
            phred = round(-10 * math.log10(prob / len(fingerprint_quals)))
            phred_scores[phred] += 1
    for i, count in enumerate(phred_scores):
        print(f"{i:2}\t{count:10,}")

