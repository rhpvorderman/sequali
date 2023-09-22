import io
from pathlib import Path

import pytest

from sequali import BamParser

import xopen

DATA = Path(__file__).parent / "data"
SIMPLE_BAM = DATA / "simple.unaligned.bam"
RAW_BAM = DATA / "simple.raw.bam"  # No BGZF format blocks.
ALIGNED_BAM = DATA / ("project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_1_NA12878.bwa."
                      "markDuplicates.bam")
COMPLETE_RECORD_WITH_HEADER = RAW_BAM.read_bytes()[:115]
COMPLETE_HEADER = COMPLETE_RECORD_WITH_HEADER[:54]


def test_bam_parser():
    with xopen.xopen(SIMPLE_BAM, "rb") as fileobj:
        parser = BamParser(fileobj)
        record_arrays = list(parser)
    assert len(record_arrays) == 1
    records = record_arrays[0]
    assert records[0].name() == "Myheader"
    assert records[0].sequence() == "GATTACA"
    assert records[0].qualities() == "HHHHHHH"
    assert records[1].name() == "AnotherHeader"
    assert records[1].sequence() == "ACATTAG"
    assert records[1].qualities() == "KKKKKKK"
    assert records[2].name() == "YetAnotherHeader"
    assert records[2].sequence() == "AAAATTTT"
    assert records[2].qualities() == "XKLLCCCC"
    assert len(records) == 3


def test_bam_parser_sequences_with_extended_header():
    with xopen.xopen(ALIGNED_BAM, "rb") as fileobj:
        parser = BamParser(fileobj)
        record_arrays = list(parser)
    assert len(record_arrays) == 1
    records = record_arrays[0]
    assert len(records) == 3
    assert records[0].name() == "HWI-D00119:50:H7AP8ADXX:1:1104:8519:18990"
    assert (records[0].sequence() ==
            "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCT"
            "GGGGGGTATGCACGCGATAGCATTGCGAGACGCTGG")
    assert (records[0].qualities() ==
            "CCCFFFFFHFFHHJIJJIJGGJJJJJJJJJJJJJIGHIIEHIJJJJJJIJJJJIBGGIIIHIIII"
            "HHHHDD;9CCDEDDDDDDDDDDEDDDDDDDDDDDDD")
    assert records[1].name() == "HWI-D00119:50:H7AP8ADXX:1:2104:18479:82511"
    assert (records[1].sequence() ==
            "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCT"
            "GGGGGGTATGCACGCGATAGCATTGCGAGACGCTGG")
    assert (records[1].qualities() ==
            "CCCFFFFFHFFHHJJJJIJJJJIIJJJJJGIJJJJGIJJJJJJJJGJIJJIJJJGHIJJJJJJJI"
            "HHHHDD@>CDDEDDDDDDDDDDEDDCDDDDD?BBD9")
    assert records[2].name() == "HWI-D00119:50:H7AP8ADXX:1:2105:7076:23015"
    assert (records[2].sequence() ==
            "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCT"
            "GGGGGGTATGCACGCGATAGCATTGCGAGACGCTGG")
    assert (records[2].qualities() ==
            "@@CFFFDFGFHHHJIIJIJIJJJJJJJIIJJJJIIJIJFIIJJJJIIIGIJJJJDHIJIIJIJJJ"
            "HHGGCB>BDDDDDDDDDDDBDDEDDDDDDDDDDDDD")


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
    with open(RAW_BAM, "rb") as fileobj:
        bam_parser = BamParser(fileobj)
        # No BAM magic, no size of header and no size of references
        assert bam_parser.header == COMPLETE_HEADER[8:-4]
    pass


@pytest.mark.parametrize("end", range(len(COMPLETE_HEADER)))
def test_parse_truncated_header(end):
    truncated_header = COMPLETE_HEADER[:end]
    fileobj = io.BytesIO(truncated_header)
    with pytest.raises(EOFError) as error:
        BamParser(fileobj)
    error.match("runcated BAM")


def test_not_a_bam():
    with pytest.raises(ValueError) as error:
        BamParser(io.BytesIO(b"@my header"))
    error.match("not a BAM")


def test_bam_parser_not_binary_error():
    with xopen.xopen(SIMPLE_BAM, "rt", encoding="latin-1") as fileobj:
        with pytest.raises(TypeError) as error:
            _ = BamParser(fileobj)
        error.match("binary IO")


def test_bam_parser_too_small_buffer():
    with pytest.raises(ValueError) as error:
        BamParser(io.BytesIO(), initial_buffersize=3)
    error.match("at least 4")
    error.match("3")


@pytest.mark.parametrize("initial_buffersize", [4, 8, 10, 20, 40])
def test_small_initial_buffer(initial_buffersize):
    with xopen.xopen(SIMPLE_BAM, "rb") as fileobj:
        parser = BamParser(fileobj, initial_buffersize=initial_buffersize)
        assert len(list(parser)) == 3
