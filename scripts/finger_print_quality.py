import argparse
import sys

import dnaio

QUAL_TO_PHRED = tuple(10 ** (-(i - 33) / 10)  for i in range(128))


def fingerprint_sequence_original(sequence: str):
    if len(sequence) < 32:
        return sequence
    return sequence[:16] + sequence[-16:]


def new_fingerprint(sequence: str,
                    front_length: int,
                    back_length: int,
                    front_offset: int,
                    back_offset: int):
    fingerprint_length = front_length + back_length
    if len(sequence) < fingerprint_length:
        return sequence

    remainder = len(sequence) - fingerprint_length
    front_offset = min(remainder // 2, front_offset)
    back_offset = min(remainder // 2, back_offset)
    return (sequence[front_offset: front_offset + front_length] +
            sequence[-(back_offset + back_length):-back_offset])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fastq")
    parser.add_argument("--front-length", default=8, type=int)
    parser.add_argument("--back-length", default=8, type=int)
    parser.add_argument("--front-offset", default=64, type=int)
    parser.add_argument("--back-offset", default=64, type=int)
    args = parser.parse_args()
    front_length = args.front_length
    back_length = args.back_length
    fingerprint_length = front_length + back_length
    expected_errors = [0 for _ in range(fingerprint_length + 1)]
    with dnaio.open(args.fastq, mode="r", open_threads=1) as reader:
        for read in reader:  # type: dnaio.SequenceRecord
            fingerprint_quals = new_fingerprint(
                read.qualities,
                front_length=front_length,
                back_length=back_length,
                front_offset=args.front_offset,
                back_offset=args.back_offset,
            )
            prob = 0.0
            for q in fingerprint_quals.encode("ascii"):
                prob += QUAL_TO_PHRED[q]
            expected_errors[round(prob)] += 1
    for i, count in enumerate(expected_errors):
        print(f"{i:2}\t{count:10,}")

