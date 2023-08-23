# Copyright (C) 2023 Leiden University Medical Center
# This file is part of sequali
#
# sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with sequali.  If not, see <https://www.gnu.org/licenses/

import gzip
import os
import string
from typing import BinaryIO, Callable, Iterator, List, Optional, Tuple

import tqdm

from ._qc import FastqRecordArrayView


class ProgressUpdater():
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

    def __init__(self, filename, filereader: BinaryIO):
        self.previous_file_pos = 0
        self.current_processed_bytes = 0
        self.progress_update_every = 1024 * 1024 * 10
        self.next_update_at = self.progress_update_every
        total: Optional[int] = os.stat(filename).st_size
        if isinstance(filereader, gzip.GzipFile):
            self._get_position = filereader.fileobj.tell
        elif filereader.seekable():
            self._get_position = filereader.tell
        else:
            self._get_position = lambda: self.current_processed_bytes
            total = None
        self.tqdm = tqdm.tqdm(
            desc=f"Processing {os.path.basename(filename)}",
            unit="iB", unit_scale=True, unit_divisor=1024,
            total=total,
            smoothing=0.01,  # Much less erratic than default 0.3
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do one last update to ensure the entire progress bar is full
        self.tqdm.update(self._get_position() - self.previous_file_pos)
        self.tqdm.close()

    def update(self, record_array: FastqRecordArrayView):
        self.current_processed_bytes += len(record_array.obj)
        if self.current_processed_bytes > self.next_update_at:
            self.next_update_at += self.progress_update_every
            current_position = self._get_position()
            self.tqdm.update(current_position - self.previous_file_pos)
            self.previous_file_pos = current_position


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


def guess_sequencing_technology(data: bytes) -> Optional[str]:
    """
    Guess sequencing technology from a block of binary data at the start of
    the file.
    :param data: a block of data
    :return:
    """
    if data[0] == b"@":
        # This is A FASTQ file.
        header_end = data.find(b"\n")
        if header_end == -1:
            header_end = None
        header = data[1:header_end].decode("ascii")
        name, metadata = header.split(maxsplit=1)  # type: str, str
        # https://help.basespace.illumina.com/files-used-by-basespace/fastq-files
        if name.count(':') == 6 and metadata.count(':') == 3:
            _, is_filtered, _, _ = metadata.split(':')
            if is_filtered in ("Y", "N"):
                return "illumina"
        # Nanopore works with UUIDs such as
        # 35eb0273-89e2-4093-98ed-d81cbdafcac7
        if name.count("-") == 4:
            hexdigits = set(string.hexdigits)
            parts = name.split('-')
            hexadecimal = all(set(part).issubset(hexdigits) for part in parts)
            correct_lengths = all(len(part) == expected for part, expected in
                                  zip(parts, (8, 4, 4, 4, 12)))
            has_ch = any(meta.startswith("ch=") for meta in metadata.split())
            has_start_time = any(meta.startswith("start_time=")
                                 for meta in metadata.split())
            if (hexadecimal and correct_lengths and has_ch and has_start_time):
                return "nanopore"
        return None
