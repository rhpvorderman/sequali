import collections
import itertools
import math

import pytest

from sequali import FastqRecordView, SequenceDuplication


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_sequence_duplication():
    max_unique_sequences = 100_000
    seqdup = SequenceDuplication(max_unique_sequences=max_unique_sequences,
                                 sample_every=1)
    # Create unique sequences by using all combinations of ACGT for the amount
    # of letters that is necessary to completely saturate the maximum unique
    # sequences
    number_of_letters = math.ceil(math.log(max_unique_sequences) / math.log(4))
    for combo in itertools.product(*(("ACGT",) * number_of_letters)):
        sequence = "".join(combo) + (31 - number_of_letters) * "A"
        read = FastqRecordView("name", sequence, "H" * len(sequence))
        seqdup.add_read(read)
    assert seqdup.number_of_sequences == 4 ** number_of_letters
    sequence_counts = seqdup.sequence_counts()
    assert len(sequence_counts) == max_unique_sequences
    assert seqdup.max_unique_sequences == max_unique_sequences
    for sequence, count in sequence_counts.items():
        assert len(sequence) == seqdup.sequence_length
        assert count == 1
    duplicated_read = FastqRecordView("name", 31 * "A",  31 * "A")
    seqdup.add_read(duplicated_read)
    sequence_counts = seqdup.sequence_counts()
    assert sequence_counts[31 * "A"] == 2


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
    seqdup = SequenceDuplication(sample_every=1)
    for i in range(100):
        seqdup.add_read(view_from_sequence("A" * 31))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C" * 31))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G" * 31))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T" * 31))
    seqdup.add_read(view_from_sequence("C" * 30 + "A"))
    for i in range(100_000 - (100 + 200 + 2000 + 10 + 1)):
        # Count up to 100_000 to get nice fractions for all the sequences
        seqdup.add_read(view_from_sequence("A" * 30 + "C"))
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.001)
    assert overrepresented[0][2] == "A" * 30 + "C"
    assert overrepresented[1][2] == "C" * 31
    assert overrepresented[1][0] == 2200
    assert overrepresented[2][2] == "A" * 31
    assert overrepresented[2][1] == 110 / 100_000
    # Assert no other sequences recorded as overrepresented.
    assert len(overrepresented) == 3
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.00001)
    assert overrepresented[-1][2] == "C" * 30 + "A"
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.00001,
        min_threshold=2,
    )
    assert overrepresented[-1][2] == "A" * 31
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.1,
        min_threshold=2,
        max_threshold=1000,
    )
    assert len(overrepresented) == 2
    assert overrepresented[1][2] == "C" * 31


def test_sequence_duplication_duplication_counts():
    seqdup = SequenceDuplication(sample_every=1)
    for i in range(100):
        seqdup.add_read(view_from_sequence("A" * 31))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C" * 31))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G" * 31))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T" * 31))
    seqdup.add_read(view_from_sequence("GATTACA" * 5))
    seqdup.add_read(view_from_sequence("TACCAGA" * 5))
    for i in range(50_000):
        seqdup.add_read(view_from_sequence("CAT" * 11))
    dupcounts = collections.Counter(seqdup.duplication_counts())
    assert dupcounts[0] == 0
    assert dupcounts[1] == 4
    assert dupcounts[2200] == 1
    assert dupcounts[110] == 1
    assert dupcounts[50_000] == 2
    assert sum(dupcounts.values()) == 8


def test_sequence_duplication_case_insensitive():
    seqdup = SequenceDuplication(sample_every=1)
    seqdup.add_read(view_from_sequence("aaTTaca" * 5))
    seqdup.add_read(view_from_sequence("AAttACA" * 5))
    seqcounts = seqdup.sequence_counts()
    assert seqdup.number_of_sequences == 2
    assert seqdup.total_fragments == 4
    assert seqdup.collected_unique_sequences == 2
    assert len(seqcounts) == 2
    assert seqcounts[("AATTACA" * 5)[:31]] == 2
    assert seqcounts[("AATTACA" * 5)[-31:]] == 2
