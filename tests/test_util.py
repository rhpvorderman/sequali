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

from pathlib import Path

import pytest

from sequali.util import (fasta_parser,
                          fastq_header_is_illumina, fastq_header_is_nanopore,
                          guess_sequencing_technology_from_bam_header,
                          sequence_names_match)

from .test_fastq_record_array import NAME_MATCH_TESTS


DATA = Path(__file__).parent / "data"
SAM = DATA / ("project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_1_NA12878.bwa."
              "markDuplicates.sam")


@pytest.mark.parametrize(["header", "is_illumina"], (
    ("SIM:1:FCX:1:15:6329:1045 1:N:0:2", True),
    ("SIM:1:FCX:1:15:6329:1045 1:Y:0:2", True),
    ("SIM:1:FCX:1:15:6329:1045", True),
))
def test_fastq_header_is_illumina(header, is_illumina):
    assert fastq_header_is_illumina(header) == is_illumina


@pytest.mark.parametrize(["header", "is_nanopore"], (
    ("SIM:1:FCX:1:15:6329:1045 1:N:0:2", False),
    ("SIM:1:FCX:1:15:6329:1045 1:Y:0:2", False),
    ("35eb0273-89e2-4093-98ed-d81cbdafcac7 "
     "runid=1d18d9e9682449156d70520e06571f01c4e6d2d8 sampleid=GM24185_1 "
     "read=41492 ch=2628 start_time=2019-01-26T18:52:46Z", True),
    ("4def3027-c3be-41d9-b4c1-9571b308cbdf\t"
     "ch:i:974\tst:Z:2023-06-09T11:01:33.207+00:00", True)
))
def test_fastq_header_is_nanopore(header, is_nanopore):
    assert fastq_header_is_nanopore(header) == is_nanopore


def test_guess_from_bam():
    sam_file = SAM.read_bytes()
    assert guess_sequencing_technology_from_bam_header(sam_file) == "illumina"


def test_fasta_parser():
    fasta = DATA / "test.fasta"
    assert list(fasta_parser(str(fasta))) == [
        ("A simple long sequence", "A" * 100),
        ("A simple multiline sequence", "AAAA"),
        ("Shortseq", "gattaca")
    ]


@pytest.mark.parametrize(["name1", "name2", "expected"], NAME_MATCH_TESTS)
def test_sequence_names_match(name1, name2, expected):
    assert sequence_names_match(name1, name2) is expected
