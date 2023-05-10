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

import array
from typing import Dict, Iterable, List, Optional, Tuple

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

class FastqRecordView:
    def __init__(self, __name, __sequence, _qualities) -> None: ...

class FastqParser:
    def __init__(self, fileobj, initial_buffer_size = 8192): ...
    def __iter__(self) -> FastqParser: ...
    def __next__(self) -> FastqRecordView: ...

class QCMetrics:
    number_of_reads: int
    max_length: int
    def __init__(self): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def count_table(self) -> array.ArrayType: ...
    def gc_content(self) -> array.ArrayType: ...
    def phred_scores(self) -> array.ArrayType: ...

class AdapterCounter:
    number_of_sequences: int
    max_length: int
    adapters: Tuple[str, ...]
    def __init__(self, __adapters: Iterable[str]): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def get_counts(self) -> List[Tuple[str, array.ArrayType]]: ...

class PerTileQuality:
    max_length: int 
    number_of_reads: int 
    skipped_reason: Optional[str]
    def __init__(self): ...
    def add_read(self, __read: FastqRecordView) -> None: ... 
    def get_tile_averages(self) -> List[Tuple[int, List[float]]]: ...

class SequenceDuplication:
    number_of_sequences: int

    def __init__(self): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def sequence_counts(self) -> Dict[str, int]: ...
    def overrepresented_sequences(self, threshold: float = 0.001) -> List[Tuple[float, str]]: ...
    def duplication_counts(self, max_count = 50_000) -> array.ArrayType: ...