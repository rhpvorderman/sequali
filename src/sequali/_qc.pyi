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

import array
import sys
from typing import Dict, Iterable, Iterator, List, SupportsIndex, Optional, Tuple

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
DEFAULT_END_ANCHOR_LENGTH: int
DEFAULT_MAX_UNIQUE_FRAGMENTS: int
DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS: int
DEFAULT_FRAGMENT_LENGTH: int
DEFAULT_UNIQUE_SAMPLE_EVERY: int
DEFAULT_BASES_FROM_START: int
DEFAULT_BASES_FROM_END: int
DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH: int
DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH: int
DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET: int
DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET: int
INSERT_SIZE_MAX_ADAPTER_STORE_SIZE: int


class FastqRecordView:
    obj: bytes
    def __init__(self, name: str, sequence: str, qualities: str, 
                 tags: Optional[bytes] = None) -> None: ...
    def name(self) -> str: ...
    def sequence(self) -> str: ...
    def qualities(self) -> str: ...
    def tags(self) -> bytes: ...

class FastqRecordArrayView:
    obj: bytes
    def __init__(self, view_items: Iterable[FastqRecordView]) -> None: ...
    def __getitem__(self, index: SupportsIndex) -> FastqRecordView: ...
    def __len__(self) -> int: ...
    def is_mate(self, other: FastqRecordArrayView): ...

class FastqParser:
    def __init__(self, fileobj, initial_buffersize = 128 * 1024): ...
    def __iter__(self) -> FastqParser: ...
    def __next__(self) -> FastqRecordArrayView: ...
    def read(self, number_of_records: int) -> FastqRecordArrayView: ...

class BamParser:
    header: bytes
    def __init__(self, fileobj, initial_buffersize = 96 * 1024): ...
    def __iter__(self) -> BamParser: ...
    def __next__(self) -> FastqRecordArrayView: ...

class QCMetrics:
    number_of_reads: int
    max_length: int
    end_anchor_length: int
    def __init__(self, end_anchor_length: int = DEFAULT_END_ANCHOR_LENGTH): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def base_count_table(self) -> array.ArrayType: ...
    def phred_count_table(self) -> array.ArrayType: ...
    def end_anchored_base_count_table(self) -> array.ArrayType: ...
    def end_anchored_phred_count_table(self) -> array.ArrayType: ...
    def gc_content(self) -> array.ArrayType: ...
    def phred_scores(self) -> array.ArrayType: ...

class AdapterCounter:
    number_of_sequences: int
    max_length: int
    adapters: Tuple[str, ...]
    def __init__(self, __adapters: Iterable[str]): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def get_counts(self) -> List[Tuple[str, array.ArrayType, array.ArrayType]]: ...

class PerTileQuality:
    max_length: int 
    number_of_reads: int 
    skipped_reason: Optional[str]
    def __init__(self): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def get_tile_counts(self) -> List[Tuple[int, List[float], List[int]]]: ...

class OverrepresentedSequences:
    number_of_sequences: int
    sampled_sequences: int
    collected_unique_fragments: int
    max_unique_fragments: int
    fragment_length: int
    sample_every: int
    total_fragments: int

    def __init__(self,
                 max_unique_fragments: int = DEFAULT_MAX_UNIQUE_FRAGMENTS,
                 fragment_length: int = DEFAULT_FRAGMENT_LENGTH,
                 sample_every: int = DEFAULT_UNIQUE_SAMPLE_EVERY,
                 bases_from_start: int = DEFAULT_BASES_FROM_START,
                 bases_from_end = DEFAULT_BASES_FROM_END,
    ): ...
    def add_read(self, __read: FastqRecordView) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def sequence_counts(self) -> Dict[str, int]: ...
    def overrepresented_sequences(self, 
                                  threshold_fraction: float = 0.0001,
                                  min_threshold: int = 1,
                                  max_threshold: int = sys.maxsize,
                                  ) -> List[Tuple[int, float, str]]: ...

class DedupEstimator:
    _modulo_bits: int 
    _hash_table_size: int 
    tracked_sequences: int
    front_sequence_length: int 
    back_sequence_length: int 
    front_sequence_offset: int 
    back_sequence_offset: int

    def __init__(
            self,
            max_stored_fingerprints: int = DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS,
            *,
            front_sequence_length: int = DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH,
            back_sequence_length: int = DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH,
            front_sequence_offset: int = DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET,
            back_sequence_offset: int = DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET,
    ): ...
    def add_sequence(self, __sequence: str) -> None: ...
    def add_sequence_pair(self, __sequence1: str, __sequence2: str) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def add_record_array_pair(self, 
                              __record_array1: FastqRecordArrayView,
                              __record_array2: FastqRecordArrayView,
                              ) -> None: ...
    def duplication_counts(self) -> array.ArrayType: ...

class NanoporeReadInfo:
    start_time: int
    channel_id: int
    length: int
    cumulative_error_rate: float
    duration: float
    parent_id_hash: int

class NanoStats:
    number_of_reads: int
    skipped_reason: Optional[str]
    minimum_time: int
    maximum_time: int
    def add_read(self, __read: FastqRecordView) -> None: ...
    def add_record_array(self, __record_array: FastqRecordArrayView) -> None: ...
    def nano_info_iterator(self) -> Iterator[NanoporeReadInfo]: ...


class InsertSizeMetrics:
    total_reads: int
    number_of_adapters_read1: int
    number_of_adapters_read2: int

    def __init__(self): ...
    def add_sequence_pair(self, __sequence1: str, __sequence2: str) -> None: ...
    def add_record_array_pair(self,
                              __record_array1: FastqRecordArrayView,
                              __record_array2: FastqRecordArrayView,
                              ) -> None: ...
    def insert_sizes(self) -> array.ArrayType: ...
    def adapters_read1(self) -> List[Tuple[str, int]]: ...
    def adapters_read2(self) -> List[Tuple[str, int]]: ...
