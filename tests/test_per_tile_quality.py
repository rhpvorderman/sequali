import dnaio
import pytest

from sequali import PerTileQuality


def test_per_tile_quality():
    read = dnaio.SequenceRecord(
        "SIM:1:FCX:1:15:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    assert ptq.number_of_reads == 1
    assert ptq.max_length == 4
    assert ptq.skipped_reason is None
    averages = ptq.get_tile_averages()
    assert len(averages) == 1
    tile, average_list = averages[0]
    assert tile == 15
    assert len(average_list) == 4
    assert average_list[0] == 10 ** (-32 / 10)
    assert average_list[1] == 10 ** (-33 / 10)
    assert average_list[2] == 10 ** (-34 / 10)
    assert average_list[3] == 10 ** (-35 / 10)


def test_per_tile_quality_not_sequence_record():
    ptq = PerTileQuality()
    with pytest.raises(TypeError) as error:
        ptq.add_read("FASTQ forever!")  # type ignore
    error.match("SequenceRecord")


def test_per_tile_quality_skip():
    read = dnaio.SequenceRecord(
        "SIMULATED_NAME",
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    assert ptq.number_of_reads == 0
    assert ptq.max_length == 0
    assert "SIMULATED_NAME" in ptq.skipped_reason
