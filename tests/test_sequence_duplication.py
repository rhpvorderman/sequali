import collections
import itertools
import math

import pytest

from sequali import FastqRecordView, SequenceDuplication
from sequali._qc import MAX_UNIQUE_SEQUENCES


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_sequence_duplication():
    seqdup = SequenceDuplication()
    # Create unique sequences by using all combinations of ACGT for the amount
    # of letters that is necessary to completely saturate the maximum unique
    # sequences
    number_of_letters = math.ceil(math.log(MAX_UNIQUE_SEQUENCES) / math.log(4))
    for combo in itertools.product(*(("ACGT",) * number_of_letters)):
        sequence = "".join(combo) + 91 * "A"
        read = FastqRecordView("name", sequence, "H" * len(sequence))
        seqdup.add_read(read)
    assert seqdup.number_of_sequences == 4 ** number_of_letters
    sequence_counts = seqdup.sequence_counts()
    assert len(sequence_counts) == MAX_UNIQUE_SEQUENCES
    for sequence, count in sequence_counts.items():
        assert len(sequence) == 50
        assert count == 1
    duplicated_read = FastqRecordView("name", 100 * "A", 100 * "A")
    seqdup.add_read(duplicated_read)
    assert seqdup.sequence_counts()[50 * "A"] == 2


def test_sequence_duplication_add_read_no_view():
    seqdup = SequenceDuplication()
    with pytest.raises(TypeError) as error:
        seqdup.add_read(b"ACGT")  # type: ignore
    error.match("FastqRecordView")
    error.match("bytes")


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_sequence_duplication_overrepresented_sequences_faulty_threshold(threshold):
    seqdup = SequenceDuplication()
    with pytest.raises(ValueError) as error:
        seqdup.overrepresented_sequences(threshold_fraction=threshold)
    error.match(str(threshold))
    error.match("between")
    error.match("0.0")
    error.match("1.0")


def test_sequence_duplication_overrepresented_sequences():
    seqdup = SequenceDuplication()
    for i in range(100):
        seqdup.add_read(view_from_sequence("A"))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C"))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G"))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T"))
    seqdup.add_read(view_from_sequence("GATTACA"))
    for i in range(100_000 - (100 + 200 + 2000 + 10 + 1)):
        # Count up to 100_000 to get nice fractions for all the sequences
        seqdup.add_read(view_from_sequence("CAT"))
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.001)
    assert overrepresented[0][2] == "CAT"
    assert overrepresented[1][2] == "G"
    assert overrepresented[1][0] == 2000
    assert overrepresented[2][2] == "C"
    assert overrepresented[2][0] == 200
    assert overrepresented[3][2] == "A"
    assert overrepresented[3][1] == 100 / 100_000
    # Assert no other sequences recorded as overrepresented.
    assert len(overrepresented) == 4
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.00001)
    assert overrepresented[-1][2] == "GATTACA"
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.00001,
        min_threshold=2,
    )
    assert overrepresented[-1][2] == "T"
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.1,
        min_threshold=2,
        max_threshold=1000,
    )
    assert len(overrepresented) == 2
    assert overrepresented[1][2] == "G"


def test_sequence_duplication_duplication_counts():
    seqdup = SequenceDuplication()
    for i in range(100):
        seqdup.add_read(view_from_sequence("A"))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C"))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G"))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T"))
    seqdup.add_read(view_from_sequence("GATTACA"))
    seqdup.add_read(view_from_sequence("TACCAGA"))
    for i in range(50_000):
        seqdup.add_read(view_from_sequence("CAT"))
    dupcounts = collections.Counter(seqdup.duplication_counts())
    assert dupcounts[0] == 0
    assert dupcounts[1] == 2
    assert dupcounts[10] == 1
    assert dupcounts[200] == 1
    assert dupcounts[100] == 1
    assert dupcounts[2000] == 1
    assert dupcounts[50_000] == 1
    assert sum(dupcounts.values()) == 7


def test_sequence_duplication_case_insensitive():
    seqdup = SequenceDuplication()
    seqdup.add_read(view_from_sequence("gaTTaca"))
    seqdup.add_read(view_from_sequence("GAttACA"))
    seqcounts = seqdup.sequence_counts()
    assert seqdup.number_of_sequences == 2
    assert len(seqcounts) == 1
    assert seqcounts["GATTACA"] == 2
