import json
import sys
from pathlib import Path

from sequali.__main__ import main

TEST_DATA = Path(__file__).parent / "data"


def test_simple_fastq(tmp_path):
    simple_fastq = TEST_DATA / "simple.fastq"
    sys.argv = ["", "--dir", str(tmp_path), str(simple_fastq)]
    main()
    simple_fastq_json = tmp_path / "simple.fastq.json"
    assert simple_fastq_json.exists()
    result = json.loads(simple_fastq_json.read_text())
    assert result["summary"]["maximum_length"] == 8
    assert result["summary"]["minimum_length"] == 7
    assert result["summary"]["total_gc_bases"] == 4
    assert result["summary"]["total_bases"] == 22


def test_empty_file(tmp_path):
    empty_fastq= TEST_DATA / "empty.fastq"
    sys.argv = ["", "--dir", str(tmp_path), str(empty_fastq)]
    main()
    simple_fastq_json = tmp_path / "empty.fastq.json"
    assert simple_fastq_json.exists()
    result = json.loads(simple_fastq_json.read_text())
    assert result["summary"]["maximum_length"] == 0
    assert result["summary"]["minimum_length"] == 0
    assert result["summary"]["total_gc_bases"] == 0
    assert result["summary"]["total_bases"] == 0


def test_empty_read(tmp_path):
    empty_read_fastq = TEST_DATA / "empty_read.fastq"
    sys.argv = ["", "--dir", str(tmp_path), str(empty_read_fastq)]
    main()
    simple_fastq_json = tmp_path / "empty_read.fastq.json"
    assert simple_fastq_json.exists()
    result = json.loads(simple_fastq_json.read_text())
    assert result["summary"]["maximum_length"] == 0
    assert result["summary"]["minimum_length"] == 0
    assert result["summary"]["total_gc_bases"] == 0
    assert result["summary"]["total_bases"] == 0
