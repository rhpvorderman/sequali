import typing
from pathlib import Path
from typing import Dict, Iterator, Tuple

FASTA_lINE_LENGTH = 70

ONT_BARCODES = Path(__file__).parent / "oxford_nanopore_barcodes.tsv"
ONT_NAIVE_BARCODES = Path(__file__).parent / "oxford_nanopore_native_barcodes.tsv"


def native_barcodes_lookup() -> Dict[str, Tuple[str, str]]:
    lookup: Dict[str, Tuple[str, str]] = {}
    with open(ONT_NAIVE_BARCODES) as f:
        _ = next(f)  # skip header
        for line in f:
            component, forward_sequence, reverse_sequence = line.strip().split('\t')
            lookup[component] = (forward_sequence, reverse_sequence)
    return lookup


def barcodes_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    with open(ONT_BARCODES) as f:
        _ = next(f)  # skip header
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


def all_barcodes() -> Iterator[FastaEntry]:
    native_barcode_lookup = native_barcodes_lookup()
    for i in range(1, 97):
        barcode_name = f"NB{i:02}"
        forward, reverse = native_barcode_lookup[barcode_name]
        yield FastaEntry(f"Oxford nanopore native barcode sequence, "
                         f"{barcode_name}, forward",
                         f"AAGGTTAA{forward}CAGCACCT")
        yield FastaEntry(f"Oxford nanopore native barcode sequence, "
                         f"{barcode_name}, reverse",
                         f"GGTGCTG{reverse}TTAACCTTAGCAAT")
    barcode_lookup = barcodes_lookup()
    for i in range(1, 97):
        barcode_name = f"RB{i:02}"
        barcode_seq = barcode_lookup[barcode_name]
        yield FastaEntry(f"Oxford nanopore rapid barcode sequence, "
                         f"{barcode_name}",
                         f"GCTTGGGTGTTTAACC{barcode_seq}GTTTTCGCATTTATCGTGAAAC"
                         f"GCTTTCGCGTTTTTCGTGCGCCGCTTCA")
    for i in range(1, 25):
        barcode_name = f"BP{i:02}"
        barcode_seq = barcode_lookup[barcode_name]
        yield FastaEntry(f"Oxford nanopore PCR barcode sequence, "
                         f"{barcode_name}, top strand",
                         f"ATCGCCTACCGTGA{barcode_seq}TTGCCTGTCGCTCTATCTTC")
        yield FastaEntry(f"Oxford nanopore PCR barcode sequence, "
                         f"{barcode_name}, bottom strand",
                         f"ATCGCCTACCGTGA{barcode_seq}TCTGTTGGTGCTGATATTGC")
    for i in range(1, 25):
        barcode_name = f"16S{i:02}"
        barcode_seq = barcode_lookup[barcode_name]
        # Forward primer contains a M base that can be both A and C
        yield FastaEntry(f"Oxford nanopore 16S barcode sequence, "
                         f"{barcode_name}, forward primer, "
                         f"with A at wobble position",
                         f"ATCGCCTACCGTGAC{barcode_seq}AGAGTTTGATCATGGCTCAG")
        yield FastaEntry(f"Oxford nanopore 16S barcode sequence, "
                         f"{barcode_name}, forward primer, "
                         f"with C at wobble position",
                         f"ATCGCCTACCGTGAC{barcode_seq}AGAGTTTGATCCTGGCTCAG")
        yield FastaEntry(f"Oxford nanopore 16S barcode sequence, "
                         f"{barcode_name}, reverse primer",
                         f"ATCGCCTACCGTGAC{barcode_seq}CGGTTACCTTGTTACGACTT")
    for i in range(1, 25):
        barcode_name = f"RLB{i:02}"
        barcode_seq = barcode_lookup[barcode_name]
        yield FastaEntry(f"Oxford nanopore rapid PCR barcode sequence, "
                         f"{barcode_name}",
                         f"ATCGCCTACCGTGAC{barcode_seq}CGTTTTTCGTGCGCCGCTTC")
    # Separate entry for RLB012A
    barcode_name = "RLB012A"
    barcode_seq = barcode_lookup[barcode_name]
    yield FastaEntry(f"Oxford nanopore rapid PCR barcode sequence, "
                     f"{barcode_name}",
                     f"ATCGCCTACCGTGAC{barcode_seq}CGTTTTTCGTGCGCCGCTTC")
    for i in range(1, 97):
        barcode_name = f"BC{i:02}"
        barcode_seq = barcode_lookup[barcode_name]
        yield FastaEntry(f"Oxford nanopore rapid barcode sequence, "
                         f"{barcode_name}",
                         f"GGTGCTG{barcode_seq}TTAACCT")


if __name__ == "__main__":
    with open("src/sequali/contaminants/oxford_nanopore_barcodes.fasta", "wt") as f:
        for barcode_entry in all_barcodes():
            f.write(str(barcode_entry))
