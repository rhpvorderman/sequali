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

from sequali import A, C, G, N, T
from sequali import FastqRecordView, QCMetrics
from sequali import NUMBER_OF_NUCS, NUMBER_OF_PHREDS


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_qc_metrics():
    sequence = "A" * 10 + "C" * 10 + "G" * 10 + "T" * 10 + "N" * 10
    qualities = chr(10 + 33) * 25 + chr(30 + 33) * 25
    metrics = QCMetrics()
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.max_length == len(sequence)
    assert metrics.number_of_reads == 1
    gc_content = metrics.gc_content()
    assert sum(gc_content) == 1
    assert gc_content[50] == 1
    phred_content = metrics.phred_scores()
    this_read_error = (10 ** -1) * 25 + (10 ** -3) * 25
    this_read_phred = -10 * math.log10(this_read_error / len(sequence))
    phred_index = math.floor(this_read_phred)
    assert phred_content[phred_index] == 1
    assert sum(phred_content) == 1
    phred_array = metrics.phred_count_table()
    assert len(phred_array) == len(sequence) * NUMBER_OF_PHREDS
    assert sum(phred_array[(10 // 4):len(phred_array):NUMBER_OF_PHREDS]) == 25
    assert sum(phred_array[(30 // 4):len(phred_array):NUMBER_OF_PHREDS]) == 25
    assert sum(phred_array) == len(sequence)
    for i in range(25):
        assert phred_array[(10 // 4) + NUMBER_OF_PHREDS * i] == 1
    for i in range(25, 50):
        assert phred_array[(30 // 4) + NUMBER_OF_PHREDS * i] == 1
    base_array = metrics.base_count_table()
    assert len(base_array) == len(sequence) * NUMBER_OF_NUCS
    assert sum(base_array[A: len(base_array): NUMBER_OF_NUCS]) == 10
    assert sum(base_array[C: len(base_array): NUMBER_OF_NUCS]) == 10
    assert sum(base_array[G: len(base_array): NUMBER_OF_NUCS]) == 10
    assert sum(base_array[T: len(base_array): NUMBER_OF_NUCS]) == 10
    assert sum(base_array[N: len(base_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array) == len(sequence)
    for i in range(10):
        assert base_array[A + NUMBER_OF_NUCS * i] == 1
    for i in range(10, 20):
        assert base_array[C + NUMBER_OF_NUCS * i] == 1
    for i in range(20, 30):
        assert base_array[G + NUMBER_OF_NUCS * i] == 1
    for i in range(30, 40):
        assert base_array[T + NUMBER_OF_NUCS * i] == 1
    for i in range(40, 50):
        assert base_array[N + NUMBER_OF_NUCS * i] == 1


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
