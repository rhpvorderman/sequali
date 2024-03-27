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

import itertools
import string

import pytest

from sequali._qc import DedupEstimator


def test_dedup_estimator():
    dedup_est = DedupEstimator(160)
    assert dedup_est._hash_table_size == 1 << 8
    dedup_est.add_sequence("test")
    dedup_est.add_sequence("test2")
    dedup_est.add_sequence("test3")
    dedup_est.add_sequence("test4")
    for i in range(100):
        dedup_est.add_sequence("test5")
    dupcounts = list(dedup_est.duplication_counts())
    assert len(dupcounts) == dedup_est.tracked_sequences
    dupcounts.sort()
    assert dupcounts[-1] == 100
    assert dupcounts[0] == 1


def test_dedup_estimator_switches_modulo():
    dedup_est = DedupEstimator(179)
    assert dedup_est._modulo_bits == 0
    ten_alphabets = [string.ascii_letters] * 10
    infinite_seqs = ("".join(letters) for letters in itertools.product(*ten_alphabets))
    for i, seq in zip(range(10000), infinite_seqs):
        dedup_est.add_sequence(seq)
    assert dedup_est._modulo_bits != 1
    # 179 seqs can be stored.
    # 10_000 / 179 = 56 sequences per slot. That requires 6 bits modulo,
    # selecting one in 64 sequences.
    assert dedup_est._modulo_bits == 6


@pytest.mark.parametrize(
    ["front_sequence_length",
     "front_sequence_offset",
     "back_sequence_length",
     "back_sequence_offset"],
    [
        (1, 0, 1, 0),
        (8, 64, 8, 64),
        (100000, 80000, 10000, 80000),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
    ],

)
def test_dedup_estimator_valid_settings(
    front_sequence_length,
    front_sequence_offset,
    back_sequence_length,
    back_sequence_offset,
):
    dedup_est = DedupEstimator(
        front_sequence_length=front_sequence_length,
        front_sequence_offset=front_sequence_offset,
        back_sequence_length=back_sequence_length,
        back_sequence_offset=back_sequence_offset,
    )
    dedup_est.add_sequence("test")
    dedup_est.add_sequence("test2")
    dedup_est.duplication_counts()


@pytest.mark.parametrize(
    ["parameter", "value"],
    [
        ("max_stored_fingerprints", 7),
        ("front_sequence_length", -1),
        ("back_sequence_length", -1),
        ("front_sequence_offset", -1),
        ("back_sequence_offset", -1),
    ]
)
def test_dedup_estimator_invalid_settings(parameter, value):
    kwargs = {parameter: value}
    with pytest.raises(ValueError) as e:
        DedupEstimator(**kwargs)
    assert e.match(parameter)
    assert e.match(str(value))


@pytest.mark.parametrize(
    ["front_sequence_length",
     "front_sequence_offset",
     "back_sequence_length",
     "back_sequence_offset",
     "result"
     ],
    [
        (8, 0, 8, 0, {1, }),
        (0, 0, 6, 0, {1, }),
        (1, 6, 1, 6, {6, }),
        (2, 6, 1, 6, {3, }),
        (2, 6, 2, 6, {2, 1, }),
        (1, 0, 0, 0, {1, }),
        (0, 0, 1, 0, {6, }),
    ],
)
def test_dedup_estimator_offsets_and_lengths(
        front_sequence_length,
        front_sequence_offset,
        back_sequence_length,
        back_sequence_offset,
        result,
):
    input_sequences = [
        "123456AC TA123451",
        "234561AC AA234561",
        "345612AC TA345611",
        "456123AG AA456121",
        "561234AG TA561231",
        "612345AG AA612341",
    ]
    dedup_est = DedupEstimator(
        front_sequence_offset=front_sequence_offset,
        front_sequence_length=front_sequence_length,
        back_sequence_length=back_sequence_length,
        back_sequence_offset=back_sequence_offset,
        max_stored_fingerprints=100,
    )
    for sequence in input_sequences:
        dedup_est.add_sequence(sequence)
    assert set(dedup_est.duplication_counts()) == result
