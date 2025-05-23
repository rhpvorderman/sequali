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

import json
import sys
import zipfile
from pathlib import Path

import pytest

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
    assert result["sequence_length_distribution"]["n50"] == 7
    assert result["sequence_length_distribution"]["n90"] == 7


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


def test_empty_file_paired(tmp_path):
    empty_fastq = TEST_DATA / "empty.fastq"
    sys.argv = ["", "--dir", str(tmp_path), str(empty_fastq), str(empty_fastq)]
    main()
    simple_fastq_json = tmp_path / "empty.fastq.json"
    assert simple_fastq_json.exists()
    result = json.loads(simple_fastq_json.read_text())
    assert result["summary_read2"]["maximum_length"] == 0
    assert result["summary_read2"]["minimum_length"] == 0
    assert result["summary_read2"]["total_gc_bases"] == 0
    assert result["summary_read2"]["total_bases"] == 0


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


def test_empty_read_paired(tmp_path):
    empty_read_fastq = TEST_DATA / "empty_read.fastq"
    sys.argv = ["", "--dir", str(tmp_path), str(empty_read_fastq),
                str(empty_read_fastq)]
    main()
    simple_fastq_json = tmp_path / "empty_read.fastq.json"
    assert simple_fastq_json.exists()
    result = json.loads(simple_fastq_json.read_text())
    assert result["summary_read2"]["maximum_length"] == 0
    assert result["summary_read2"]["minimum_length"] == 0
    assert result["summary_read2"]["total_gc_bases"] == 0
    assert result["summary_read2"]["total_bases"] == 0


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


def test_nanopore_reads(tmp_path):
    # Nanopore reads from GM24385_1.fastq.gz from GIAB data.
    # https://github.com/genome-in-a-bottle/giab_data_indexes/blob/ade38024efc7f3db5be27ee270e4fb9537a97a1a/AshkenazimTrio/sequence.index.AJtrio_UCSC_ONT_UL_Promethion_03312019.HG002#L2
    nanopore_fastq = TEST_DATA / "100_nanopore_reads.fastq.gz"
    sys.argv = ["", "--dir", str(tmp_path), str(nanopore_fastq)]
    main()
    fastq_json = tmp_path / "100_nanopore_reads.fastq.gz.json"
    result = json.loads(fastq_json.read_text())
    assert result["summary"]["total_reads"] == 100
    assert result["sequence_length_distribution"]["n50"] == 59502
    assert result["sequence_length_distribution"]["n90"] == 7517


def test_dorado_nanopore_bam(tmp_path):
    # Called from ont open data. See bam header.
    nanopore_bam = TEST_DATA / "dorado_nanopore_100reads.bam"
    sys.argv = ["", "--threads", "1", "--dir", str(tmp_path), str(nanopore_bam)]
    main()
    fastq_json = tmp_path / "dorado_nanopore_100reads.bam.json"
    result = json.loads(fastq_json.read_text())
    assert result["summary"]["total_reads"] == 100


def test_single_nuc(tmp_path):
    single_nuc_fastq = (TEST_DATA / "single_nuc.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(single_nuc_fastq)]
    main()
    single_nuc_fastq_json = tmp_path / "single_nuc.fastq.json"
    assert single_nuc_fastq_json.exists()
    result = json.loads(single_nuc_fastq_json.read_text())
    assert result["summary"]["maximum_length"] == 1
    assert result["summary"]["minimum_length"] == 1
    assert result["summary"]["total_gc_bases"] == 0
    assert result["summary"]["total_bases"] == 1


def test_single_nanopore_metada(tmp_path):
    fastq = (TEST_DATA / "single_nanopore_metadata.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq)]
    main()
    fastq_json = tmp_path / "single_nanopore_metadata.fastq.json"
    assert fastq_json.exists()


def test_empty_nanopore_metada(tmp_path):
    fastq = (TEST_DATA / "empty_nanopore_metadata.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq)]
    main()
    fastq_json = tmp_path / "empty_nanopore_metadata.fastq.json"
    assert fastq_json.exists()


def test_single_illumina_metada(tmp_path):
    fastq = (TEST_DATA / "single_illumina_metadata.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq)]
    main()
    fastq_json = tmp_path / "single_illumina_metadata.fastq.json"
    assert fastq_json.exists()


def test_empty_illumina_metada(tmp_path):
    fastq = (TEST_DATA / "empty_illumina_metadata.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq)]
    main()
    fastq_json = tmp_path / "empty_illumina_metadata.fastq.json"
    assert fastq_json.exists()


def test_nanopore_disparate_dates(tmp_path):
    fastq = (TEST_DATA / "nanopore_disparate_dates.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq)]
    main()
    fastq_json = tmp_path / "nanopore_disparate_dates.fastq.json"
    assert fastq_json.exists()


def test_paired_end_sequences(tmp_path):
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R1_001.fastq.gz")
    fastq2 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    main()
    fastq_json = tmp_path / "LTB-A-BC001_S1_L003_R1_001.fastq.gz.json"
    assert fastq_json.exists()
    json_data = json.loads(fastq_json.read_text())
    assert "summary_read2" in json_data


def test_paired_end_sequences_read1_shorter(tmp_path):
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R1_001_shortened.fastq.gz")
    fastq2 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    assert error.match("R2_001.fastq.gz has more FASTQ records")


def test_paired_end_sequences_read2_shorter(tmp_path):
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R1_001.fastq.gz")
    fastq2 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001_shortened.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    assert error.match("R1_001.fastq.gz has more FASTQ records")


def test_paired_end_sequence_bam_format_read1(tmp_path):
    fastq1 = (
        TEST_DATA /
        "project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_1_NA12878.bwa.markDuplicates.bam"
    )
    fastq2 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    error.match("Paired end mode is only supported for FASTQ files")


def test_paired_end_sequence_bam_format_read2(tmp_path):
    fastq2 = (
        TEST_DATA /
        "project.NIST_NIST7035_H7AP8ADXX_TAAGGCGA_1_NA12878.bwa.markDuplicates.bam"
    )
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    error.match("Paired end mode is only supported for FASTQ files")


def test_paired_end_sequences_mismatching_sequencing_technologies(tmp_path):
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R1_001.fastq.gz")
    fastq2 = (TEST_DATA / "simple.fastq")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    error.match("Mismatching sequencing technologies")
    error.match("illumina")
    error.match("None")


def test_paired_end_names_mismatching(tmp_path):
    fastq1 = (TEST_DATA / "LTB-A-BC001_S1_L003_R1_001_names_changed.fastq.gz")
    fastq2 = (TEST_DATA / "LTB-A-BC001_S1_L003_R2_001.fastq.gz")
    sys.argv = ["", "--dir", str(tmp_path), str(fastq1), str(fastq2)]
    with pytest.raises(RuntimeError) as error:
        main()
    error.match("ismatching names")
    error.match("D00360:76:C672MANXX:3:1101:5469:2124X 1:N:0:1")
    error.match("D00360:76:C672MANXX:3:1101:5469:2124 2:N:0:1")


def test_version_command(capsys):
    sys.argv = ["", "--version"]
    with pytest.raises(SystemExit):
        main()
    result = capsys.readouterr()
    import sequali
    assert result.out.replace("\n", "") == sequali.__version__


@pytest.mark.parametrize("testfiles", [
    [str(TEST_DATA / "simple.fastq")],
    [str(TEST_DATA / "LTB-A-BC001_S1_L003_R1_001_shortened.fastq.gz"),
     str(TEST_DATA / "LTB-A-BC001_S1_L003_R2_001_shortened.fastq.gz")],
    [str(TEST_DATA / "100_nanopore_reads.fastq.gz")]
])
def test_images_zip(tmp_path, testfiles):
    images_zip = tmp_path / "images.zip"
    sys.argv = ["", "--dir", str(tmp_path), *testfiles,
                "--images-zip", str(images_zip)]
    main()

    assert images_zip.exists()
    with zipfile.ZipFile(images_zip) as z:
        for info in z.infolist():
            assert info.filename.endswith(".svg")
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
