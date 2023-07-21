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

import sys
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple, Union

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


def reverse_complement(sequence: str):
    return sequence.encode("ascii").translate(COMPLEMENT_TABLE)[::-1].decode("ascii")


def canonical_kmers(sequence: str, k: int):
    if k % 2 == 0:
        raise ValueError(f"K must be uneven, got {k}")
    # Encoding to bytes makes translating faster
    seq = sequence.encode("ascii")
    complement_table = COMPLEMENT_TABLE
    upper_table = UPPER_TABLE
    canonical_set = set()
    for i in range(len(seq) + 1 - k):
        kmer = seq[i:i+k].translate(upper_table)
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
        self._hash = hash(self.name) ^ hash(self.kmers)

    def __hash__(self):
        return self._hash


def create_sequence_index(
        names_and_sequences: Iterable[Tuple[str, str]],
        k: int = DEFAULT_K,
        ) -> Dict[str, Union[List[SequenceIdentifier], SequenceIdentifier]]:
    sequence_index: Dict[str, Union[List[SequenceIdentifier], SequenceIdentifier]] = {}
    for name, sequence in names_and_sequences:
        kmers = canonical_kmers(sequence, k)
        seq_identifier = SequenceIdentifier(name, kmers)
        # Store one sequence identifier (most common) or alternatively a list
        # of multiple sequence identifiers. This is a bit convoluted, but it
        # saves a lot of time as memory does not need to be allocated for a
        # lot of one-item lists.
        for kmer in kmers:
            if kmer in sequence_index:
                prev_entry = sequence_index[kmer]
                if isinstance(prev_entry, list):
                    sequence_index[kmer].append(seq_identifier)  # type: ignore
                else:
                    sequence_index[kmer] = [seq_identifier, prev_entry]
            else:
                sequence_index[kmer] = seq_identifier
    return sequence_index


def identify_sequence(
        sequence: str,
        sequence_index: Dict[str, Union[List[SequenceIdentifier], SequenceIdentifier]],
        k: int = DEFAULT_K) -> Tuple[int, int, str]:
    kmers = canonical_kmers(sequence, k)
    candidates: Set[SequenceIdentifier] = set()
    for kmer in kmers:
        matched = sequence_index.get(kmer, [])
        if isinstance(matched, list):
            candidates.update(matched)
        else:
            candidates.add(matched)
    most_matches = 0
    best_match = "No match"
    best_match_kmers = sys.maxsize
    for candidate in candidates:
        matches = len(kmers & candidate.kmers)
        if matches > most_matches:
            best_match_kmers = len(candidate.kmers)
            most_matches = matches
            best_match = candidate.name
    return most_matches, min(len(kmers), best_match_kmers), best_match
