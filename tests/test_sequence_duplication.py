import itertools

import pytest

from sequali import SequenceDuplication


def test_sequence_duplication():
    seqdup = SequenceDuplication()
    # Create unique sequences by using all combinations of ACGT for 9 letters
    # This gives 4 ** 9 or 262144 unique sequences
    for combo in itertools.product(*(("ACGT",) * 9)):
        seqdup.add_sequence("".join(combo) + 91 * "A")
    assert seqdup.number_of_sequences == 4 ** 9
    assert len(seqdup.sequence_counts) == 100_000
    for sequence, count in seqdup.sequence_counts.items():
        assert len(sequence) == 50
        assert count == 1
    seqdup.add_sequence(100 * "A")  # Should already exist
    assert seqdup.sequence_counts[50 * "A"] == 2


def test_sequence_duplication_add_sequence_no_string():
    seqdup = SequenceDuplication()
    with pytest.raises(TypeError) as error:
        seqdup.add_sequence(b"ACGT")  # type: ignore
    error.match("str")
    error.match("bytes")


def test_sequence_duplication_add_sequence_not_ascii():
    seqdup = SequenceDuplication()
    with pytest.raises(ValueError) as error:
        seqdup.add_sequence("ÄCGT")
    error.match("ASCII")
