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
