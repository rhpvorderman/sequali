import collections
import itertools
import os
import sys

import pytest

from sequali import FastqRecordView, SequenceDuplication


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_sequence_duplication():
    seqdup = SequenceDuplication()
    # Create unique sequences by using all combinations of ACGT for 9 letters
    # This gives 4 ** 9 or 262144 unique sequences
    for combo in itertools.product(*(("ACGT",) * 9)):
        sequence = "".join(combo) + 91 * "A"
        read = FastqRecordView("name", sequence, "H" * len(sequence))
        seqdup.add_read(read)
    assert seqdup.number_of_sequences == 4 ** 9
    sequence_counts = seqdup.sequence_counts()
    assert len(sequence_counts) == 100_000
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
        seqdup.overrepresented_sequences(threshold=threshold)
    error.match(str(threshold))
    error.match("between")
    error.match("0.0")
    error.match("1.0")


# TODO: investigate the memory error on windows. On linux the maximum memory
# TODO: used does not exceed 70MB for the entire test suite.
@pytest.mark.skipif(sys.platform == "win32", bool(os.environ.get("CI")),
                    reason="Windows somehow gets a memory error in CI.")
def test_sequence_duplication_overrepresented_sequences():
    seqdup = SequenceDuplication()
    for i in range(100):
        seqdup.add_read(view_from_sequence("mildly overrepresented"))
    for i in range(200):
        seqdup.add_read(view_from_sequence("slightly overrepresented"))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("Blatantly overrepresented"))
    for i in range(10):
        seqdup.add_read(view_from_sequence("not overrepresented"))
    seqdup.add_read(view_from_sequence("truly unique"))
    for i in range(100_000 - (100 + 200 + 2000 + 10 + 1)):
        # Count up to 100_000 to get nice fractions for all the sequences
        seqdup.add_read(view_from_sequence("SPAM"))
    overrepresented = seqdup.overrepresented_sequences(threshold=0.001)
    assert overrepresented[0][2] == "SPAM"
    assert overrepresented[1][2] == "Blatantly overrepresented"
    assert overrepresented[1][0] == 2000
    assert overrepresented[2][2] == "slightly overrepresented"
    assert overrepresented[2][0] == 200
    assert overrepresented[3][2] == "mildly overrepresented"
    assert overrepresented[3][1] == 100 / 100_000
    # Assert no other sequences recorded as overrepresented.
    assert len(overrepresented) == 4


def test_sequence_duplication_duplication_counts():
    seqdup = SequenceDuplication()
    for i in range(100):
        seqdup.add_read(view_from_sequence("mildly overrepresented"))
    for i in range(200):
        seqdup.add_read(view_from_sequence("slightly overrepresented"))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("Blatantly overrepresented"))
    for i in range(10):
        seqdup.add_read(view_from_sequence("not overrepresented"))
    seqdup.add_read(view_from_sequence("truly unique"))
    seqdup.add_read(view_from_sequence("another unique"))
    for i in range(50_000):
        seqdup.add_read(view_from_sequence("SPAM"))
    dupcounts = collections.Counter(seqdup.duplication_counts())
    assert dupcounts[0] == 0
    assert dupcounts[1] == 2
    assert dupcounts[10] == 1
    assert dupcounts[200] == 1
    assert dupcounts[100] == 1
    assert dupcounts[2000] == 1
    assert dupcounts[50_000] == 1
    assert sum(dupcounts.values()) == 7
