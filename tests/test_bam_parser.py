import io
import re
from pathlib import Path

import pytest

from sequali import BamParser

import xopen

DATA = Path(__file__).parent / "data"
SIMPLE_BAM = DATA / "simple.unaligned.bam"
RAW_BAM = DATA / "simple.raw.bam"  # No BGZF format blocks.


def test_bam_parser():
    with open(SIMPLE_BAM, "rb") as fileobj:
        parser = BamParser(fileobj)
        record_arrays = list(parser)
    assert len(record_arrays) == 1
    records = record_arrays[0]
    assert records[0].name() == "Myheader/1"
    assert records[0].sequence() == "GATTACA"
    assert records[0].qualities() == "HHHHHHH"
    assert records[1].name() == "AnotherHeader/1"
    assert records[1].sequence() == "ACATTAG"
    assert records[1].qualities() == "KKKKKKK"
    assert records[2].name() == "YetAnotherHeader/1"
    assert records[2].sequence() == "AAAATTTT"
    assert records[2].qualities() == "XKLLCCCC"
    assert len(records) == 3


COMPLETE_RECORD_WITH_HEADER = RAW_BAM.read_bytes()[:115]
COMPLETE_HEADER = COMPLETE_RECORD_WITH_HEADER[:54]


@pytest.mark.parametrize("end", range(len(COMPLETE_HEADER) + 1,
                                      len(COMPLETE_RECORD_WITH_HEADER)))
def test_truncated_record(end: int):
    truncated_record = COMPLETE_RECORD_WITH_HEADER[:end]
    fileobj = io.BytesIO(truncated_record)
    parser = BamParser(fileobj)
    with pytest.raises(EOFError) as error:
        list(parser)
    error.match("ncomplete record")


def test_parse_header():
    # TODO
    pass


def test_fastq_parser_not_binary_error():
    with xopen.xopen(SIMPLE_BAM, "rt", encoding="latin-1") as fileobj:
        parser = BamParser(fileobj)
        with pytest.raises(TypeError) as error:
            list(parser)
        error.match("binary IO")
        error.match(repr(fileobj))


def test_fastq_parser_too_small_buffer():
    with pytest.raises(ValueError) as error:
        BamParser(io.BytesIO(), initial_buffersize=0)
    error.match("at least 1")
    error.match("0")


@pytest.mark.parametrize("initial_buffersize", [1, 2, 4, 8, 10, 20, 40])
def test_small_initial_buffer(initial_buffersize):
    with xopen.xopen(SIMPLE_BAM, "rb") as fileobj:
        parser = BamParser(fileobj, initial_buffersize=initial_buffersize)
        assert len(list(parser)) == 3
