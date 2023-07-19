import pytest

from sequali.sequence_identification import canonical_kmers, reverse_complement


@pytest.mark.parametrize(["sequence", "rev_complement"], [
    ("ATA", "TAT"),
    ("ATCG", "CGAT"),
    ("ANTCG", "CGANT")
])
def test_reverse_complement(sequence: str, rev_complement: str):
    assert reverse_complement(sequence) == rev_complement


def test_canonical_kmers():
    assert canonical_kmers("GATTACA", 3) == {"ATC", "AAT", "TAA", "GTA", "ACA"}
    assert canonical_kmers("gattaca", 3) == \
           canonical_kmers(reverse_complement("GATTACA"), 3)
