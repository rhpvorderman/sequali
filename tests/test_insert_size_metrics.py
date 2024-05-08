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

import pytest

from sequali._qc import INSERT_SIZE_MAX_ADAPTER_STORE_SIZE, InsertSizeMetrics

ILLUMINA_ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
ILLUMINA_ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"


@pytest.mark.parametrize(["sequence1", "sequence2", "insert_size"], [
    (
        "ATATATATATATATAT",
        "ATATATATATATATAT",
        16
    ),
    (
        "ATATATATATATATATNNNNNNNNNN",
        "ATATATATATATATATNNNNNNNNNN",
        16
    ),
    (
        "NNNNNNNNNNATATATATATATATAT",
        "ATATATATATATATATNNNNNNNNNN",
        26
    ),
    (
        "ACGTTGCAGCTATCGA" + ILLUMINA_ADAPTER_R1,
        "TCGATAGCTGCAACGT" + ILLUMINA_ADAPTER_R2,
        16
    ),
    (
        "GTACACGTTGCAGCTATCGA" + ILLUMINA_ADAPTER_R1,
        "TCGATAGCTGCAACGTGTAC" + ILLUMINA_ADAPTER_R2,
        20
    ),
    (
        "GTACACGTTGCAGCTATCGA" + ILLUMINA_ADAPTER_R1,
        "tcgatagctgcaacgtgtac" + ILLUMINA_ADAPTER_R2,
        20
    ),
    (
        "GTACACGTTGCAGCTATCGA" + ILLUMINA_ADAPTER_R1,
        "tcGatagCTgcaAcgtGtac" + ILLUMINA_ADAPTER_R2,
        20
    ),
])
def test_insert_size_metrics(sequence1, sequence2, insert_size):
    insert_size_metrics = InsertSizeMetrics()
    insert_size_metrics.add_sequence_pair(
        sequence1,
        sequence2
    )
    assert insert_size_metrics.insert_sizes()[insert_size] == 1
    adapter_1 = sequence1[insert_size:][:INSERT_SIZE_MAX_ADAPTER_STORE_SIZE]
    if adapter_1:
        assert dict(insert_size_metrics.adapters_read1()).get(adapter_1) == 1
        assert insert_size_metrics.number_of_adapters_read1 == 1
    else:
        assert insert_size_metrics.number_of_adapters_read1 == 0
    adapter_2 = sequence2[insert_size:][:INSERT_SIZE_MAX_ADAPTER_STORE_SIZE]
    if adapter_2:
        assert dict(insert_size_metrics.adapters_read2()).get(adapter_2) == 1
        assert insert_size_metrics.number_of_adapters_read2 == 1
    else:
        assert insert_size_metrics.number_of_adapters_read2 == 0
