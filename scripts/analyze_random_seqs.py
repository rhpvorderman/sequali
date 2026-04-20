#!/usr/bin/env python3

from pathlib import Path
import statistics

import dnaio

if __name__ == "__main__":
    fastq_file = Path(__file__).parent.parent / "tests" / "data" / "random_seqs.fastq"
    number_of_records = 0
    record_lengths = []
    a_counts = []
    c_counts = []
    g_counts = []
    t_counts = []
    with dnaio.open(fastq_file) as fastq:
        for record in fastq:
            record_lengths.append(len(record))
            number_of_records += 1
            seq = record.sequence.upper()
            a_counts.append(seq.count("A"))
            c_counts.append(seq.count("C"))
            g_counts.append(seq.count("G"))
            t_counts.append(seq.count("T"))
    print(f"Median length: {statistics.median(record_lengths)}")
    print(f"Mean length: {statistics.mean(record_lengths)}")
    a = sum(a_counts)
    c = sum(c_counts)
    g = sum(g_counts)
    t = sum(t_counts)
    print(f"Total bases: {sum(record_lengths)}")
    print(f"total GC bases: {g + c}")
    print(f"GC content: {(g + c) / (a + g + c + t)}")
