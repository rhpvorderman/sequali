import sys

import dnaio

import collections

from sequali.report_modules import DuplicationCounts


def fastq_file_to_hashes(fastq_file):
    with dnaio.open(sys.argv[1], open_threads=1) as reader:
        for read in reader:
            length = len(read)
            if length < 16:
                h = hash(read.sequence)
            else:
                h = hash(read.sequence[:16]) ^ (length >> 6) ^ hash(read.sequence[-16:])
            yield h


if __name__ == "__main__":
    counter = collections.Counter(fastq_file_to_hashes(sys.argv[1]))
    dupcounter = collections.Counter(counter.values())
    print(dict(sorted(dupcounter.items())))
    estimated_fractions = DuplicationCounts.estimated_counts_to_fractions(dupcounter.items())
    print(estimated_fractions)