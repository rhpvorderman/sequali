import sys

import dnaio

import collections

from sequali.report_modules import DuplicationCounts

if __name__ == "__main__":
    counter = collections.Counter()
    with dnaio.open(sys.argv[1]) as reader:
        for read in reader:
            length = len(read)
            if length < 16:
                h = hash(read.sequence)
            else:
                h = hash(read.sequence[:16]) ^ length ^ hash(read.sequence[-16:])
            counter.update((h,))
    dupcounter = collections.Counter()
    dupcounter.update(counter.values())
    estimated_fractions = DuplicationCounts.estimated_counts_to_fractions(dupcounter.items())
    print(estimated_fractions)