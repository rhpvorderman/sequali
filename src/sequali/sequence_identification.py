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

import collections
from typing import Dict, FrozenSet, Iterable, Optional, Set, Tuple

DEFAULT_K = 13


def create_upper_table():
    upper_table = bytearray(b"N" * 256)
    for c in "acgtACGT":
        upper_table[ord(c)] = ord(c.upper())
    return bytes(upper_table)


def create_complement_table():
    complement_table = bytearray(create_upper_table())
    for c, complement in zip("acgtACGT", "TGCATGCA"):
        complement_table[ord(c)] = ord(complement)
    return bytes(complement_table)


COMPLEMENT_TABLE = create_complement_table()
UPPER_TABLE = create_upper_table()


def canonical_kmers(sequence: str, k: int):
    if k % 2 == 0:
        raise ValueError(f"K must be uneven, got {k}")
    # Encoding to bytes makes translating faster
    sequence = sequence.encode("ascii")
    complement_table = COMPLEMENT_TABLE
    upper_table = UPPER_TABLE
    canonical_set = set()
    for i in range(len(sequence) + 1 - k):
        kmer = sequence[i:i+k].translate(upper_table)
        revcomp = kmer.translate(complement_table)[::-1]
        if revcomp < kmer:
            canonical_set.add(revcomp.decode("ascii"))
        else:
            canonical_set.add(kmer.decode("ascii"))
    return canonical_set


class SequenceIdentifier:
    name: str
    kmers: FrozenSet[str]
    _hash: int
    __slots__ = ["name", "kmers", "_hash"]

    def __init__(self, name: str, kmers: Iterable[str]):
        self.name = name
        self.kmers = frozenset(kmers)
        self._hash = hash(name) ^ hash(kmers)

    def __hash__(self):
        return self._hash


def create_sequence_index(names_and_sequences: Iterable[Tuple[str, str]],
                          k: int = DEFAULT_K,
                          ) -> Dict[str, Set[SequenceIdentifier]]:
    sequence_index = collections.defaultdict(set)
    for name, sequence in names_and_sequences:
        kmers = canonical_kmers(sequence, k)
        seq_identifier = SequenceIdentifier(name, kmers)
        for kmer in kmers:
            sequence_index[kmer].add(seq_identifier)
    return sequence_index


def identify_sequence(sequence: str) -> Optional[str]:
    return None
