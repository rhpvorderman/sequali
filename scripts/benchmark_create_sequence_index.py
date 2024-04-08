import resource
import sys
import gc

from sequali.sequence_identification import (DEFAULT_CONTAMINANTS_FILES,
                                             create_sequence_index)
from sequali.util import fasta_parser


def default_sequences():
    for f in DEFAULT_CONTAMINANTS_FILES:
        yield from fasta_parser(f)


if __name__ == "__main__":
    prior_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sequence_index = create_sequence_index(default_sequences())
    resource_usage = resource.getrusage(resource.RUSAGE_SELF)
    total_key_usage = 0
    total_container_usage = 0
    for key, value in sequence_index.items():
        total_key_usage += sys.getsizeof(key)
        if not isinstance(value, str):
            total_container_usage += sys.getsizeof(value)
    print(f"dict memory usage:\t{sys.getsizeof(sequence_index) / (1024 * 1024):.2f} MiB")
    print(f"key memory usage:\t{total_key_usage / (1024 * 1024):.2f} MiB")
    print(f"container memory usage:\t{total_container_usage / (1024 * 1024):.2f} MiB")
    print(f"total memory usage:\t{(resource_usage.ru_maxrss - prior_mem_usage) / 1024:.2f} MiB")
    print(f"total kmers stored:\t{len(sequence_index):,}")
    print(f"{resource_usage.ru_utime + resource_usage.ru_stime :.2f} seconds")
