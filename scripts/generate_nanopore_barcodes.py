from pathlib import Path
from typing import Dict, Tuple

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

