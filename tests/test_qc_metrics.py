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

import math

import pytest

from sequali import A, C, G, N, T
from sequali import FastqRecordView, QCMetrics
from sequali import NUMBER_OF_NUCS, NUMBER_OF_PHREDS

from .test_fastq_parser import DATA, simple_fastq_parser


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


PHREDS_TO_ERRORS = [
    10 ** -((x - 33) / 10) for x in range(128)
]


def expected_errors(qualities: str):
    return sum(PHREDS_TO_ERRORS[ord(c)] for c in qualities)


def base_to_index(base: str) -> int:
    assert len(base) == 1
    return {
        "A": A,
        "C": C,
        "G": G,
        "T": T,
    }.get(base.upper(), N)


@pytest.mark.parametrize(
    ["sequence", "qualities", "end_anchor_length"],
    [
        (
            "A" * 10 + "C" * 10 + "G" * 10 + "T" * 10 + "N" * 10,
            chr(10 + 33) * 25 + chr(30 + 33) * 25,
            15
        ),
        (
                "A" * 10 + "C" * 10 + "G" * 10 + "T" * 10 + "N" * 10,
                chr(10 + 33) * 25 + chr(30 + 33) * 25,
                100
        ),
        (
                "A" * 10 + "C" * 10 + "G" * 10 + "T" * 10 + "N" * 10,
                chr(10 + 33) * 25 + chr(30 + 33) * 25,
                50
        ),
        (
                "A" * 50,
                chr(1 + 33) * 25 + chr(5 + 33) * 25,
                100
        ),
        *[
            (x.sequence, x.qualities, 100)
            for x in simple_fastq_parser(str(DATA / "simple.fastq"))
        ],
        *[
            pytest.param(x.sequence, x.qualities, 100, id=f"nanopore_{i}")
            for i, x in enumerate(
                simple_fastq_parser(str(DATA / "100_nanopore_reads.fastq.gz"))
            )
        ]
    ],
)
def test_qc_metrics(sequence, qualities, end_anchor_length):
    metrics = QCMetrics(end_anchor_length=end_anchor_length)
    assert metrics.end_anchor_length == end_anchor_length
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.max_length == len(sequence)
    assert metrics.number_of_reads == 1
    gc_content = metrics.gc_content()
    assert sum(gc_content) == 1
    at_count = sequence.count('A') + sequence.count('T')
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content_index = round((gc_count * 100) / (at_count + gc_count))
    assert gc_content[gc_content_index] == 1
    phred_content = metrics.phred_scores()
    this_read_error = expected_errors(qualities)
    this_read_phred = -10 * math.log10(this_read_error / len(sequence))
    phred_index = math.floor(this_read_phred)
    assert phred_content[phred_index] == 1
    assert sum(phred_content) == 1
    phred_array = metrics.phred_count_table()
    assert len(phred_array) == len(sequence) * NUMBER_OF_PHREDS
    assert sum(phred_array) == len(sequence)
    for i, char in enumerate(qualities):
        phred = ord(char) - 33
        phred_index = min(phred // 4, 11)
        assert phred_array[phred_index + NUMBER_OF_PHREDS * i] == 1
    base_array = metrics.base_count_table()
    assert len(base_array) == len(sequence) * NUMBER_OF_NUCS
    assert sum(base_array[A: len(base_array): NUMBER_OF_NUCS]) == sequence.count('A')
    assert sum(base_array[C: len(base_array): NUMBER_OF_NUCS]) == sequence.count('C')
    assert sum(base_array[G: len(base_array): NUMBER_OF_NUCS]) == sequence.count('G')
    assert sum(base_array[T: len(base_array): NUMBER_OF_NUCS]) == sequence.count('T')
    assert sum(base_array[N: len(base_array): NUMBER_OF_NUCS]) == sequence.count('N')
    assert sum(phred_array) == len(sequence)
    for i, nuc in enumerate(sequence):
        assert base_array[i * NUMBER_OF_NUCS + base_to_index(nuc)] == 1
    end_anchored_bases = metrics.end_anchored_base_count_table()
    end_anchored_phreds = metrics.end_anchored_phred_count_table()
    end_anchor_length = metrics.end_anchor_length
    assert len(end_anchored_bases) == end_anchor_length * NUMBER_OF_NUCS
    end_sequence = sequence[max(len(sequence)-end_anchor_length, 0):]
    end_phreds = qualities[max(len(sequence)-end_anchor_length, 0):]
    end_offset = max(end_anchor_length - len(sequence), 0)
    for i, base in enumerate(end_sequence):
        assert end_anchored_bases[
                   (end_offset + i) * NUMBER_OF_NUCS + base_to_index(base)
               ] == 1
    for i, phred in enumerate(end_phreds):
        phred_value = ord(phred) - 33
        phred_index = min(phred_value // 4, 11)
        assert end_anchored_phreds[
                   (end_offset + i) * NUMBER_OF_PHREDS + phred_index] == 1


def test_long_sequence():
    metrics = QCMetrics()

    sequence = 4096 * 'A' + 4096 * 'C'
    qualities = (2048 * chr(00 + 33) +
                 2048 * chr(10 + 33) +
                 2048 * chr(20 + 33) +
                 2048 * chr(30 + 33))
    errors = (2048 * (10 ** -0) +
              2048 * (10 ** -1) +
              2048 * (10 ** -2) +
              2048 * (10 ** -3))
    phred = -10 * math.log10(errors / 8192)
    floored_phred = math.floor(phred)
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.phred_scores()[floored_phred] == 1
    assert metrics.gc_content()[50] == 1


def test_average_long_quality():
    metrics = QCMetrics()
    sequence = 20_000_000 * "A"
    # After creating a big error float of 1000.0 we start adding very small
    # probabilities of 1 in 100_000. If the counting is too imprecise, the
    # rounding errors will lead to an incorrect average score for
    # the entire read.
    qualities = 1000 * chr(0 + 33) + 19_999_000 * chr(50 + 33)
    error_rate = (1 + 19999 * 10 ** -5) / 20000
    phred = -10 * math.log10(error_rate)
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.phred_scores()[math.floor(phred)] == 1
