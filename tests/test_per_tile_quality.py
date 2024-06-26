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

from sequali import FastqRecordView, PerTileQuality


def test_per_tile_quality():
    read = FastqRecordView(
        "SIM:1:FCX:1:15:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    assert ptq.number_of_reads == 1
    assert ptq.max_length == 4
    assert ptq.skipped_reason is None
    counts = ptq.get_tile_counts()
    assert len(counts) == 1
    tile, sum_list, count_list = counts[0]
    assert tile == 15
    assert len(sum_list) == 4
    assert sum_list[0] == 10 ** (-32 / 10)
    assert sum_list[1] == 10 ** (-33 / 10)
    assert sum_list[2] == 10 ** (-34 / 10)
    assert sum_list[3] == 10 ** (-35 / 10)
    assert count_list == [1, 1, 1, 1]


@pytest.mark.parametrize("tile_id", list(range(100)) + [1234, 99239])
def test_tile_parse_correct(tile_id):
    read = FastqRecordView(
        f"SIM:1:FCX:1:{tile_id}:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    averages = ptq.get_tile_counts()
    assert len(averages) == 1
    tile, sum_list, count_list = averages[0]
    assert tile == tile_id


def test_per_tile_quality_not_view():
    ptq = PerTileQuality()
    with pytest.raises(TypeError) as error:
        ptq.add_read("FASTQ forever!")  # type ignore
    error.match("FastqRecordView")


@pytest.mark.parametrize("header", [
    "SIMULATED_NAME",
    "SIM:1:FCX:1::6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",  # tile empty
    "SIM:1:FCX:1:abc:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",  # tile not a number
    "SIM:1:FCX:1:0x1a3:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",  # number not decimal
    "SIM:1:FCX:1",  # truncated before tile id
    "SIM:1:FCX:1:1045",  # truncated after tile id
])
def test_per_tile_quality_skip(header):
    read = FastqRecordView(
        header,
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    assert ptq.number_of_reads == 0
    assert ptq.max_length == 0
    assert header in ptq.skipped_reason
