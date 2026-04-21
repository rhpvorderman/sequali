#!/usr/bin/env python3
import math
import statistics
from pathlib import Path

import dnaio

PHRED_TO_QUAL = [10 ** (-i / 10) for i in range(94)]


def average_quality(seq: dnaio.SequenceRecord):
    quality = 0.0
    for phred in seq.qualities.encode('ascii'):
        quality += PHRED_TO_QUAL[phred - 33]
    if len(seq) == 0:
        return 0
    return -10 * math.log10(quality / len(seq))


if __name__ == "__main__":
    fastq_file = Path(__file__).parent.parent / "tests" / "data" / "random_seqs.fastq"
    number_of_records = 0
    record_lengths = []
    a_counts = []
    c_counts = []
    g_counts = []
    t_counts = []
    average_qualities = [0 for _ in range(94)]
    gc_contents = [0 for _ in range(101)]
    with dnaio.open(fastq_file) as fastq:
        for record in fastq:
            record_lengths.append(len(record))
            number_of_records += 1
            if len(record) == 0:
                continue
            seq = record.sequence.upper()
            a_count = seq.count("A")
            c_count = seq.count("C")
            g_count = seq.count("G")
            t_count = seq.count("T")
            a_counts.append(a_count)
            c_counts.append(c_count)
            g_counts.append(g_count)
            t_counts.append(t_count)
            av_quality = average_quality(record)
            total = (a_count + c_count + g_count + t_count)
            gc_count = c_count + g_count
            gc_content = (gc_count * 100) / total
            gc_contents[round(gc_content)] += 1
            average_qualities[math.floor(av_quality)] += 1
    print(f"Median length: {statistics.median(record_lengths)}")
    print(f"Mean length: {statistics.mean(record_lengths)}")
    a = sum(a_counts)
    c = sum(c_counts)
    g = sum(g_counts)
    t = sum(t_counts)
    print(f"Total bases: {sum(record_lengths)}")
    print(f"total GC bases: {g + c}")
    print(f"GC content: {(g + c) / (a + g + c + t)}")
    print(average_qualities)
    print(gc_contents)
