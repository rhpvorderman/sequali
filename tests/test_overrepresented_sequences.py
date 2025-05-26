# Copyright (C) 2023 Leiden University Medical Center
# This file is part of Sequali
#
# Sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Sequali.  If not, see <https://www.gnu.org/licenses/
import itertools
import math
import warnings

import pytest

from sequali import FastqRecordView, OverrepresentedSequences


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_overrepresented_sequences():
    max_unique_fragments = 100_000
    fragment_length = 31
    seqdup = OverrepresentedSequences(
        max_unique_fragments=max_unique_fragments,
        fragment_length=fragment_length,
        sample_every=1
    )
    # Create unique sequences by using all combinations of ACGT for the amount
    # of letters that is necessary to completely saturate the maximum unique
    # sequences
    number_of_letters = math.ceil(math.log(max_unique_fragments) / math.log(4))
    for combo in itertools.product(*(("ACGT",) * number_of_letters)):
        sequence = "".join(combo) + (fragment_length - number_of_letters) * "A"
        read = FastqRecordView("name", sequence, "H" * len(sequence))
        seqdup.add_read(read)
    assert seqdup.number_of_sequences == 4 ** number_of_letters
    sequence_counts = seqdup.sequence_counts()
    assert len(sequence_counts) == max_unique_fragments
    assert (seqdup.max_unique_fragments == max_unique_fragments)
    for sequence, count in sequence_counts.items():
        assert len(sequence) == seqdup.fragment_length
        assert count == 1
    duplicated_read = FastqRecordView("name", fragment_length * "A",
                                      fragment_length * "A")
    seqdup.add_read(duplicated_read)
    sequence_counts = seqdup.sequence_counts()
    assert sequence_counts[fragment_length * "A"] == 2


def test_overrepresented_sequences_add_read_no_view():
    seqs = OverrepresentedSequences()
    with pytest.raises(TypeError) as error:
        seqs.add_read(b"ACGT")  # type: ignore
    error.match("FastqRecordView")
    error.match("bytes")


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_overrepresented_sequences_overrepresented_sequences_faulty_threshold(
        threshold):
    seqs = OverrepresentedSequences()
    with pytest.raises(ValueError) as error:
        seqs.overrepresented_sequences(threshold_fraction=threshold)
    error.match(str(threshold))
    error.match("between")
    error.match("0.0")
    error.match("1.0")


def test_overrepresented_sequences_overrepresented_sequences():
    fragment_length = 31
    seqs = OverrepresentedSequences(sample_every=1, fragment_length=fragment_length)
    for i in range(100):
        seqs.add_read(view_from_sequence("A" * fragment_length))
    for i in range(200):
        seqs.add_read(view_from_sequence("C" * fragment_length))
    for i in range(2000):
        seqs.add_read(view_from_sequence("G" * fragment_length))
    for i in range(10):
        seqs.add_read(view_from_sequence("T" * fragment_length))
    seqs.add_read(view_from_sequence("C" * (fragment_length - 1) + "A"))
    for i in range(100_000 - (100 + 200 + 2000 + 10 + 1)):
        # Count up to 100_000 to get nice fractions for all the sequences
        seqs.add_read(view_from_sequence("A" * (fragment_length - 1) + "C"))
    overrepresented = seqs.overrepresented_sequences(threshold_fraction=0.001)
    assert overrepresented[0][2] == "A" * (fragment_length - 1) + "C"
    assert overrepresented[1][2] == "C" * fragment_length
    assert overrepresented[1][0] == 2200
    assert overrepresented[2][2] == "A" * fragment_length
    assert overrepresented[2][1] == 110 / 100_000
    # Assert no other sequences recorded as overrepresented.
    assert len(overrepresented) == 3
    overrepresented = seqs.overrepresented_sequences(threshold_fraction=0.00001)
    assert overrepresented[-1][2] == "C" * (fragment_length - 1) + "A"
    overrepresented = seqs.overrepresented_sequences(
        threshold_fraction=0.00001,
        min_threshold=2,
    )
    assert overrepresented[-1][2] == "A" * fragment_length
    overrepresented = seqs.overrepresented_sequences(
        threshold_fraction=0.1,
        min_threshold=2,
        max_threshold=1000,
    )
    assert len(overrepresented) == 2
    assert overrepresented[1][2] == "C" * fragment_length


def test_overrepresented_sequences_case_insensitive():
    fragment_length = 31
    seqs = OverrepresentedSequences(fragment_length=fragment_length, sample_every=1)
    seqs.add_read(view_from_sequence("aaTTaca" * 5))
    seqs.add_read(view_from_sequence("AAttACA" * 5))
    seqcounts = seqs.sequence_counts()
    assert seqs.number_of_sequences == 2
    assert seqs.total_fragments == 4
    assert seqs.collected_unique_fragments == 2
    assert len(seqcounts) == 2
    assert seqcounts[("AATTACA" * 5)[:fragment_length]] == 2
    assert seqcounts[("AATTACA" * 5)[-fragment_length:]] == 2


@pytest.mark.parametrize("divisor", list(range(1, 21)))
def test_overrepresented_sequences_sampling_rate(divisor):
    seqs = OverrepresentedSequences(sample_every=divisor)
    read = view_from_sequence("AAAA")
    number_of_sequences = 10_000
    for i in range(number_of_sequences):
        seqs.add_read(read)
    assert seqs.number_of_sequences == number_of_sequences
    assert seqs.sampled_sequences == (number_of_sequences + divisor - 1) // divisor


@pytest.mark.parametrize(["sequence", "result"], [
    ("GATTACAGATTACA", {"ATC": 1, "GTA": 1, "AGA": 1, "ACA": 1, "AAT": 1}),
    ("GATTACAAA", {"ATC": 1, "GTA": 1, "AAA": 1}),
    ("GA", {}),
    ("GATT", {"ATC": 1, "AAT": 1}),
    # Fragments that are duplicated in the sequence should only be recorded once.
    ("GATTACGATTAC", {"ATC": 1, "GTA": 1}),
])
def test_overrepresented_sequences_all_fragments(sequence, result):
    seqs = OverrepresentedSequences(fragment_length=3, sample_every=1)
    seqs.add_read(view_from_sequence(sequence))
    seq_counts = seqs.sequence_counts()
    assert seq_counts == result


def test_very_short_sequence():
    # With 32 byte load this will overflow the used memory.
    seqs = OverrepresentedSequences(fragment_length=3, sample_every=1)
    seqs.add_read(view_from_sequence("ACT"))
    assert seqs.sequence_counts() == {"ACT": 1}


def test_non_iupac_warning():
    seqs = OverrepresentedSequences(fragment_length=3, sample_every=1)
    with pytest.warns(UserWarning, match="KKK"):
        seqs.add_read(view_from_sequence("KKK"))


def test_valid_does_not_warn_for_n():
    seqs = OverrepresentedSequences(fragment_length=3, sample_every=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seqs.add_read(view_from_sequence("ACGTN"))
    # N does lead to a sample not being loaded.
    assert seqs.sampled_sequences == 1


@pytest.mark.parametrize(
    ["bases_from_start", "bases_from_end", "result"], [
        (0, 0, ()),
        (1, 1, ("AAC", "CAA")),
        (2, 2, ("AAC", "CAA")),
        (3, 3, ("AAC", "CAA")),
        (4, 4, ("AAC", "CAA", "CCG", "GCC")),
        (1, 0, ("AAC",)),
        (0, 1, ("CAA",)),
        (100, 100, ("AAA", "AAC", "CAA", "CCG", "GCC")),
        (-1, -1, ("AAA", "AAC", "CAA", "CCG", "GCC")),
    ]
)
def test_overrepresented_sequences_sample_from_begin_and_end(
        bases_from_start, bases_from_end, result):
    seqs = OverrepresentedSequences(
        fragment_length=3,
        sample_every=1,
        bases_from_start=bases_from_start,
        bases_from_end=bases_from_end
    )
    seqs.add_read(view_from_sequence("AACCGGTTTTGGCCAA"))
    overrepresented = [x[2] for x in seqs.overrepresented_sequences(min_threshold=1)]
    overrepresented.sort()
    assert tuple(overrepresented) == result
