import argparse
import sys

import dnaio

QUAL_TO_PHRED = tuple(10 ** (-(i - 33) / 10)  for i in range(128))


def fingerprint_sequence_original(sequence: str):
    if len(sequence) < 32:
        return sequence
    return sequence[:16] + sequence[-16:]


def new_fingerprint(sequence: str, fingerprint_length=32, max_offset=32):
    fingerprint_part_length = fingerprint_length // 2
    if len(sequence) < fingerprint_length:
        return sequence
    remainder = len(sequence) - fingerprint_length
    offset = max(remainder // 2, max_offset)
    return sequence[offset: offset + fingerprint_part_length] + sequence[-(offset + fingerprint_part_length):-offset]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fastq")
    parser.add_argument("--fingerprint-length", nargs="?", default=32, type=int)
    parser.add_argument("--offset", nargs="?", default=32, type=int)
    args = parser.parse_args()
    offset = args.offset
    fingerprint_length = args.fingerprint_length
    expected_errors = [0 for _ in range(fingerprint_length + 1)]
    with dnaio.open(args.fastq, mode="r", open_threads=1) as reader:
        for read in reader:  # type: dnaio.SequenceRecord
            fingerprint_quals = new_fingerprint(read.qualities, fingerprint_length, offset)
            prob = 0.0
            for q in fingerprint_quals.encode("ascii"):
                prob += QUAL_TO_PHRED[q]
            expected_errors[round(prob)] += 1
    for i, count in enumerate(expected_errors):
        print(f"{i:2}\t{count:10,}")

