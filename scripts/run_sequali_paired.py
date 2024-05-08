import argparse
import os
import subprocess
import sys
from typing import Iterator, Tuple


def find_fastq_pairs(fastq_dir) -> Iterator[Tuple[str, str]]:
    for entry in os.scandir(fastq_dir):
        if entry.is_dir():
            yield from find_fastq_pairs(entry.path)
        if entry.is_file():
            if entry.name.endswith("R1_001.fastq.gz"):
                r1 = entry.path
                r2 = entry.path.replace("R1_001.fastq.gz", "R2_001.fastq.gz")
                yield r1, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fastq_dir")
    parser.add_argument("report_dir")
    args = parser.parse_args()
    for r1, r2 in find_fastq_pairs(args.fastq_dir):
        subprocess.run(["sequali", "--outdir", args.report_dir, r1, r2],
                       stderr=sys.stderr, stdout=sys.stdout, check=True)
