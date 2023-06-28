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
    averages = ptq.get_tile_averages()
    assert len(averages) == 1
    tile, average_list = averages[0]
    assert tile == 15
    assert len(average_list) == 4
    assert average_list[0] == 10 ** (-32 / 10)
    assert average_list[1] == 10 ** (-33 / 10)
    assert average_list[2] == 10 ** (-34 / 10)
    assert average_list[3] == 10 ** (-35 / 10)


@pytest.mark.parametrize("tile_id", list(range(100)) + [1234, 99239])
def test_tile_parse_correct(tile_id):
    read = FastqRecordView(
        f"SIM:1:FCX:1:{tile_id}:6329:1045:GATTACT+GTCTTAAC 1:N:0:ATCCGA",
        "AAAA",
        "ABCD"
    )
    ptq = PerTileQuality()
    ptq.add_read(read)
    averages = ptq.get_tile_averages()
    assert len(averages) == 1
    tile, average_list = averages[0]
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
