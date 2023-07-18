import pytest

from sequali.sequence_identification import COMPLEMENT_TABLE, canonical_kmers


@pytest.mark.parametrize(["sequence", "complement"], [
    ("ATA", "TAT"),
    ("ATCG", "TAGC"),
    ("ANTCG", "TNAGC")
])
def test_complement_table(sequence: str, complement: str):
    assert sequence.encode("ascii").translate(COMPLEMENT_TABLE).decode(
        "ascii") == complement


def test_canonical_kmers():
    assert canonical_kmers("GATTACA", 3) == {"ATC", "AAT", "TAA", "GTA", "ACA"}
