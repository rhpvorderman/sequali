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
    assert records[0].tags() == b""
    assert records[1].name() == "AnotherHeader/1"
    assert records[1].sequence() == "ACATTAG"
    assert records[1].qualities() == "KKKKKKK"
    assert records[1].tags() == b""
    assert records[2].name() == "YetAnotherHeader/1"
    assert records[2].sequence() == "AAAATTTT"
    assert records[2].qualities() == "XKLLCCCC"
    assert records[2].tags() == b""
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


@pytest.mark.parametrize("number_of_records", [i for i in range(1, 101)])
def test_fastq_record_array_read(number_of_records):
    with open(DATA / "100_illumina_adapters.fastq", "rb") as fileobj:
        parser = FastqParser(fileobj)
        record_array = parser.read(number_of_records)
        second_record_array = parser.read(100)
        third_record_array = parser.read(100)
    assert len(record_array) == number_of_records
    assert len(second_record_array) == 100 - number_of_records
    assert second_record_array.obj.count(b'\n') == (100 - number_of_records) * 4
    assert len(third_record_array) == 0
    assert third_record_array.obj == b""


@pytest.mark.parametrize("buffer_size", [128, 256, 512, 1024, 2048, 128 * 1024])
def test_fastq_record_array_read_buffersizes(buffer_size):
    with open(DATA / "100_illumina_adapters.fastq", "rb") as fileobj:
        parser = FastqParser(fileobj, buffer_size)
        record_array = parser.read(20)
        second_record_array = parser.read(70)
        third_record_array = parser.read(50)
    assert len(record_array) == 20
    assert len(second_record_array) == 70
    assert len(third_record_array) == 10
    assert record_array.obj.count(b"\n") >= 20 * 4
    assert second_record_array.obj.count(b"\n") >= 70 * 4
    assert third_record_array.obj.count(b"\n") == 10 * 4
