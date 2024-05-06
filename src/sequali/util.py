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
import os
import string
from typing import (
    BinaryIO,
    Callable,
    Iterator,
    List,
    Optional,
    SupportsIndex,
    Tuple,
    Union,
)

import tqdm

import xopen

from ._qc import BamParser, FastqParser, FastqRecordArrayView


try:
    from isal.igzip_threaded import _ThreadedGzipReader
except ImportError:
    _ThreadedGzipReader = None  # type: ignore


class ProgressUpdater:
    """
    A simple wrapper to update the progressbar based on the parsed
    FastqRecordArrayView objects.

    Because tqdm requires some minor execution time, only call tqdm.update()
    every 10MiB of processed records to prevent too much time spent on
    calling the tell() functions and calling tqdm.update().
    """
    _get_position: Callable[[], int]
    previous_file_pos: int
    current_processed_bytes = 0
    progress_update_every: int
    next_update_at: int
    tqdm: tqdm.tqdm

    def __init__(self, filereader: io.BufferedReader):
        self.previous_file_pos = 0
        self.current_processed_bytes = 0
        self.progress_update_every = 1024 * 1024 * 10
        self.next_update_at = self.progress_update_every
        filename = filereader.name
        total: Optional[int] = os.stat(filename).st_size
        if filereader.seekable():
            self._get_position = filereader.tell
        else:
            self._get_position = lambda: self.current_processed_bytes
            total = None
        self.tqdm = tqdm.tqdm(
            desc=f"Processing {os.path.basename(filename)}",
            unit="iB", unit_scale=True, unit_divisor=1024,
            total=total,
            smoothing=0.05,  # Much less erratic than default 0.3
        )

    def __enter__(self):
        return self

    def close(self):
        # Do one last update to ensure the entire progress bar is full
        self.tqdm.update(self._get_position() - self.previous_file_pos)
        self.tqdm.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self, record_array: FastqRecordArrayView):
        self.current_processed_bytes += len(record_array.obj)
        if self.current_processed_bytes > self.next_update_at:
            self.next_update_at += self.progress_update_every
            current_position = self._get_position()
            self.tqdm.update(current_position - self.previous_file_pos)
            self.previous_file_pos = current_position


class NGSFile:
    filepath: str
    raw: io.BufferedReader
    file: BinaryIO
    progress: ProgressUpdater
    reader: Union[BamParser, FastqParser]
    sequencing_technology: Optional[str]
    format: str

    def __init__(self, filepath: str, threads: int = 0):
        self.filepath = filepath
        self.raw = open(filepath, "rb")  # type: ignore
        self.progress = ProgressUpdater(self.raw)
        self.file = xopen.xopen(self.raw, "rb", threads=threads)
        if filepath.endswith(".bam") or (
                hasattr(self.file, "peek") and self.file.peek(4)[:4] == b"BAM\1"):
            self.reader = BamParser(self.file)
            self.sequencing_technology = \
                guess_sequencing_technology_from_bam_header(self.reader.header)
            self.format = "BAM"
        else:
            self.reader = FastqParser(self.file)
            self.sequencing_technology = \
                guess_sequencing_technology_from_file(self.file)  # type: ignore
            self.format = "FASTQ"

    def __iter__(self):
        for record_array in self.reader:
            self.progress.update(record_array)
            yield record_array

    def read(self, number_of_records: int):
        record_array = self.reader.read(number_of_records)  # type: ignore
        self.progress.update(record_array)
        return record_array

    def close(self):
        self.progress.close()
        self.file.close()
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def fasta_parser(fasta_file: str) -> Iterator[Tuple[str, str]]:
    current_seq: List[str] = []
    name = ""
    with open(fasta_file, "rt") as fasta:
        for line in fasta:
            if line.startswith(">"):
                if current_seq:
                    yield name, "".join(current_seq)
                name = line.strip()[1:]
                current_seq = []
            else:
                current_seq.append(line.strip())
        yield name, "".join(current_seq)


def guess_sequencing_technology_from_file(fp: io.BufferedReader) -> Optional[str]:
    """
    Guess sequencing technology from a block of binary data at the start of
    the file.
    :param data: a block of data
    :return:
    """
    try:
        data = fp.peek(io.DEFAULT_BUFFER_SIZE)
    except IOError:
        return None
    if not data:
        # Empty file
        return None
    if data[0] == ord("@"):
        # This is A FASTQ file.
        header_end: Optional[SupportsIndex] = data.find(b"\n")
        if header_end == -1:
            header_end = None
        header = data[1:header_end].decode("ascii")
        if fastq_header_is_illumina(header):
            return "illumina"
        if fastq_header_is_nanopore(header):
            return "nanopore"
    return None


def fastq_header_is_illumina(header: str) -> bool:
    # Illumina header format. two parts separated by a space
    # @<instrument>:<run number>:<flowcell ID>:<lane>:<tile>:<x-pos>:<y-pos>
    # <read>:<is filtered>:<control number>:<sample number>
    # See also:
    # https://help.basespace.illumina.com/files-used-by-basespace/fastq-files
    header_parts = header.split(maxsplit=1)
    if len(header_parts) == 1:
        metadata = None
    elif len(header_parts) == 2:
        metadata = header_parts[1]
    else:
        return False
    name = header_parts[0]
    # Metadata part is not always present, but if it is, it should be tested.
    if metadata:
        if metadata.count(':') != 3:
            return False
        _, is_filtered, _, _ = metadata.split(':')
        if is_filtered not in ("Y", "N"):
            return False
    if name.count(':') == 6:
        return True
    return False


def fastq_header_is_nanopore(header: str):
    # Nanopore works with UUIDs such as
    # 35eb0273-89e2-4093-98ed-d81cbdafcac7
    # After that the metadata is several parts in the form of name=data each
    # of the parts separated by a space. 'ch' for channel and 'start_time' are
    # present in guppy called FASTQ files.
    name, *metadata = header.split()  # type: str, List[str]
    if name.count("-") == 4:
        hexdigits = set(string.hexdigits)
        parts = name.split('-')
        hexadecimal = all(set(part).issubset(hexdigits) for part in parts)
        correct_lengths = all(
            len(part) == correct_length for part, correct_length in
            zip(parts, (8, 4, 4, 4, 12)))
        # Test only for ch (no =) and for st (no art_time) so also ubam to
        # converted FASTQ files are included.
        has_ch = any(meta.startswith("ch") for meta in metadata)
        has_start_time = any(meta.startswith("st")
                             for meta in metadata)
        if hexadecimal and correct_lengths and has_ch and has_start_time:
            return True
    return False


def guess_sequencing_technology_from_bam_header(bam_header: bytes):
    header = bam_header.decode("utf-8")
    lines = header.splitlines()
    for line in lines:
        if line.startswith("@RG"):
            # Use 1: because the first split will be @RG
            fields = line.split("\t")[1:]
            for field in fields:
                # Due to timestamps containing colons, only split once.
                tag, value = field.split(":", maxsplit=1)
                if tag == "PL":
                    if value == "ONT":
                        return "nanopore"
                    elif value == "Illumina":
                        return "illumina"
    return None


def sequence_names_match(name1: str, name2: str):
    id1 = name1.split(maxsplit=1)[0]
    id2 = name2.split(maxsplit=1)[0]
    id1_last_char = id1[-1]
    id2_last_char = id2[-1]
    if (id1_last_char == "1" and id2_last_char == "2") or (
            id1_last_char == "2" and id2_last_char == "1"):
        # Do not compare the /1 or /2 part at the end of paired end reads.
        id1 = id1[:-1]
        id2 = id2[:-1]
    return id1 == id2
