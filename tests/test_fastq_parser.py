import io

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
    error.match(truncated_record.decode("ascii"))
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
