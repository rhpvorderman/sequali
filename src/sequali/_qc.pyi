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

from typing import Iterable, List, Optional, Tuple

from dnaio import SequenceRecord

TABLE_SIZE: int
NUMBER_OF_PHREDS: int
NUMBER_OF_NUCS: int
PHRED_MAX: int
A: int 
C: int
G: int 
T: int 
N: int 
MAX_SEQUENCE_SIZE: int

class QCMetrics:
    number_of_reads: int
    max_length: int
    def __init__(self): ...
    def add_read(self, __read: SequenceRecord): ...
    def count_table_view(self) -> memoryview: ...
    def gc_content_view(self) -> memoryview: ...
    def phred_scores_view(self) -> memoryview: ...

class AdapterCounter:
    number_of_sequences: int
    max_length: int
    adapters: Tuple[str, memoryview]
    def __init__(self, __adapters: Iterable[str]): ...
    def add_sequence(self, __sequence: str) -> None: ...
    def get_counts(self) -> List[Tuple[str, memoryview]]: ...
    def _get_bitmatrices(self) -> memoryview: ... 

class PerTileQuality:
    max_length: int 
    number_of_reads: int 
    skipped_reason: Optional[str]
    def __init__(self): ...
    def add_read(self, __read: SequenceRecord): ... 
    def get_tile_averages(self) -> List[Tuple[int, List[float]]]: ...
