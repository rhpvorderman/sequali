import io
import re

import pytest

from sequali import FastqParser

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
    with pytest.raises(TypeError) as error:
        list(parser)
    error.match("binary IO")
    error.match(repr(fileobj))


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