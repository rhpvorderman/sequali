# Copyright (C) 2023 Leiden University Medical Center
# This file is part of sequali
#
# sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with sequali.  If not, see <https://www.gnu.org/licenses/

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
    max_unique_fragments = 100_000
    fragment_length = 31
    seqdup = SequenceDuplication(max_unique_fragments=max_unique_fragments,
                                 fragment_length=fragment_length,
                                 sample_every=1)
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
    fragment_length = 31
    seqdup = SequenceDuplication(sample_every=1, fragment_length=fragment_length)
    for i in range(100):
        seqdup.add_read(view_from_sequence("A" * fragment_length))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C" * fragment_length))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G" * fragment_length))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T" * fragment_length))
    seqdup.add_read(view_from_sequence("C" * (fragment_length - 1) + "A"))
    for i in range(100_000 - (100 + 200 + 2000 + 10 + 1)):
        # Count up to 100_000 to get nice fractions for all the sequences
        seqdup.add_read(view_from_sequence("A" * (fragment_length - 1) + "C"))
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.001)
    assert overrepresented[0][2] == "A" * (fragment_length - 1) + "C"
    assert overrepresented[1][2] == "C" * fragment_length
    assert overrepresented[1][0] == 2200
    assert overrepresented[2][2] == "A" * fragment_length
    assert overrepresented[2][1] == 110 / 100_000
    # Assert no other sequences recorded as overrepresented.
    assert len(overrepresented) == 3
    overrepresented = seqdup.overrepresented_sequences(threshold_fraction=0.00001)
    assert overrepresented[-1][2] == "C" * (fragment_length - 1) + "A"
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.00001,
        min_threshold=2,
    )
    assert overrepresented[-1][2] == "A" * fragment_length
    overrepresented = seqdup.overrepresented_sequences(
        threshold_fraction=0.1,
        min_threshold=2,
        max_threshold=1000,
    )
    assert len(overrepresented) == 2
    assert overrepresented[1][2] == "C" * fragment_length


def test_sequence_duplication_duplication_counts():
    fragment_length = 31
    seqdup = SequenceDuplication(sample_every=1, fragment_length=fragment_length)
    for i in range(100):
        seqdup.add_read(view_from_sequence("A" * fragment_length))
    for i in range(200):
        seqdup.add_read(view_from_sequence("C" * fragment_length))
    for i in range(2000):
        seqdup.add_read(view_from_sequence("G" * fragment_length))
    for i in range(10):
        seqdup.add_read(view_from_sequence("T" * fragment_length))
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
    fragment_length = 31
    seqdup = SequenceDuplication(fragment_length=fragment_length, sample_every=1)
    seqdup.add_read(view_from_sequence("aaTTaca" * 5))
    seqdup.add_read(view_from_sequence("AAttACA" * 5))
    seqcounts = seqdup.sequence_counts()
    assert seqdup.number_of_sequences == 2
    assert seqdup.total_fragments == 4
    assert seqdup.collected_unique_fragments == 2
    assert len(seqcounts) == 2
    assert seqcounts[("AATTACA" * 5)[:fragment_length]] == 2
    assert seqcounts[("AATTACA" * 5)[-fragment_length:]] == 2


@pytest.mark.parametrize("divisor", list(range(1, 21)))
def test_sequence_duplication_sampling_rate(divisor):
    seqdup = SequenceDuplication(sample_every=divisor)
    read = view_from_sequence("AAAA")
    number_of_sequences = 10_000
    for i in range(number_of_sequences):
        seqdup.add_read(read)
    assert seqdup.number_of_sequences == number_of_sequences
    assert seqdup.sampled_sequences == (number_of_sequences + divisor - 1) // divisor
