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
    empty_fastq = TEST_DATA / "empty.fastq"
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


def test_adapters_only(tmp_path):
    adapters_fastq = TEST_DATA / "100_illumina_adapters.fastq"
    sys.argv = ["", "--dir", str(tmp_path),
                "--overrepresentation-sample-every", "1",
                str(adapters_fastq)]
    main()
    adapters_fastq_json = tmp_path / "100_illumina_adapters.fastq.json"
    assert adapters_fastq_json.exists()
    result = json.loads(adapters_fastq_json.read_text())
    assert result["summary"]["maximum_length"] == 33
    assert result["summary"]["minimum_length"] == 33
    assert result["summary"]["total_gc_bases"] == 1700
    assert result["summary"]["total_bases"] == 3300
    adapter_content_lists = result["adapter_content"]["adapter_content"]
    for adapter_name, quantities in adapter_content_lists:
        if adapter_name == "Illumina Universal Adapter":
            assert quantities == [100.0] * 33
        assert len(quantities) == 33
    overrepresented_sequences = result["overrepresented_sequences"]
    assert overrepresented_sequences["total_sequences"] == 100
    assert overrepresented_sequences["sampled_sequences"] == 100
    assert overrepresented_sequences["total_fragments"] == 200
    overrepr_seqs = overrepresented_sequences["overrepresented_sequences"]
    for d in overrepr_seqs:
        assert d["count"] == 100
        # Illumina adapter should be in the database. All kmers should match.
        assert d["most_matches"] == d["max_matches"]
        assert "Illumina" in d["best_match"]
