import io
import re
from pathlib import Path

import pytest

from sequali import FastqParser

DATA = Path(__file__).parent / "data"


def test_fastq_parser():
    with open(DATA / "simple.fastq", "rb") as fileobj:
        parser = FastqParser(fileobj)
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


WRONG_RECORDS_AND_ERRORS = [
    (b"not a record", ["Record does not start with @", "with n"]),
    (b"@correctname\nSEQ\n-\n", ["start with +", "with -"]),
    (b"@correctname\nAGA\n+\nGG\n", ["equal length"]),
]


@pytest.mark.parametrize(["record", "error_messages"], WRONG_RECORDS_AND_ERRORS)
def test_fastq_parser_wrong_fastq(record, error_messages):
    parser = FastqParser(io.BytesIO(record))
    with pytest.raises(ValueError) as error:
        next(parser)
    for message in error_messages:
        error.match(message)


COMPLETE_RECORD = b"@SOMEHEADER METADATA MOREMETADATA\nAGA\n+\nGGG\n"


@pytest.mark.parametrize("end", range(1, len(COMPLETE_RECORD)))
def test_truncated_record(end: int):
    truncated_record = COMPLETE_RECORD[:end]
    fileobj = io.BytesIO(truncated_record)
    parser = FastqParser(fileobj)
    with pytest.raises(EOFError) as error:
        list(parser)
    error.match(re.escape(truncated_record.decode("ascii")))
    error.match("ncomplete record")


def test_fastq_parser_not_binary_error():
    fileobj = io.StringIO("@Name\nAGC\n+\nHHH\n")
    parser = FastqParser(fileobj)
    with pytest.raises(AttributeError) as error:
        list(parser)
    error.match("readinto")
    error.match("io.StringIO")


def test_fastq_parser_non_ascii_input():
    fileobj = io.BytesIO("@nÄmé \nAGC\n+\nHHH\n".encode("latin-1"))
    parser = FastqParser(fileobj)
    with pytest.raises(ValueError) as error:
        list(parser)
    error.match("ASCII")
    error.match("Ä")


def test_fastq_parser_too_small_buffer():
    with pytest.raises(ValueError) as error:
        FastqParser(io.BytesIO(), initial_buffersize=0)
    error.match("at least 1")
    error.match("0")


@pytest.mark.parametrize("initial_buffersize", [1, 2, 4, 8, 10, 20, 40, 100])
def test_small_initial_buffer(initial_buffersize):
    fileobj = io.BytesIO(COMPLETE_RECORD)
    parser = FastqParser(fileobj, initial_buffersize=initial_buffersize)
    assert len(list(parser)) == 1
