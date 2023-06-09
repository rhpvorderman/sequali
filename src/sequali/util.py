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
from typing import BinaryIO, Callable, Optional

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
            total=total
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
