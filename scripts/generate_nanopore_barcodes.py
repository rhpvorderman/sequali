import typing
from pathlib import Path
from typing import Dict, Iterator, Tuple

FASTA_lINE_LENGTH = 70

ONT_BARCODES = Path(__file__).parent / "oxford_nanopore_barcodes.tsv"
ONT_NAIVE_BARCODES = Path(__file__).parent / "oxford_nanopore_naive_barcodes.tsv"


def naive_barcodes_lookup() -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}
    with open(ONT_NAIVE_BARCODES) as f:
        header = next(f)
        for line in f:
            component, forward_sequence, reverse_sequence = line.strip().split('\t')
            lookup[component] = (forward_sequence, reverse_sequence)
    return lookup


def barcodes_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with open(ONT_BARCODES) as f:
        header = next(f)
        for line in f:
            components, sequence = line.strip().split("\t")
            for component in components.split(" / "):
                lookup[component] = sequence
    return lookup


class FastaEntry(typing.NamedTuple):
    name: str
    sequence: str

    def __str__(self):
        sequence_parts = (
            self.sequence[i: i + FASTA_lINE_LENGTH]
            for i in range(0, len(self.sequence), FASTA_lINE_LENGTH)
        )
        return f">{self.name}\n" + "\n".join(sequence_parts) + "\n"
