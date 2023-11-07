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
import os
import sys
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple, Union
from sequali._qc import canonical_kmers

DEFAULT_K = 13

CONTAMINANTS_DIR = os.path.join(os.path.dirname(__file__), "contaminants")
DEFAULT_CONTAMINANTS_FILES = [f.path for f in os.scandir(CONTAMINANTS_DIR)
                              if f.name != "README"]


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


def create_sequence_index(
        names_and_sequences: Iterable[Tuple[str, str]],
        k: int = DEFAULT_K,
        ) -> Dict[int, Union[List[str], str]]:
    sequence_index: Dict[int, Union[List[str], str]] = {}
    for name, sequence in names_and_sequences:
        kmers = canonical_kmers(sequence, k)
        # Store one sequence identifier (most common) or alternatively a list
        # of multiple sequence identifiers. This is a bit convoluted, but it
        # saves a lot of time as memory does not need to be allocated for a
        # lot of one-item lists.
        for kmer in kmers:
            if kmer in sequence_index:
                prev_entry = sequence_index[kmer]
                if isinstance(prev_entry, list):
                    sequence_index[kmer].append(name)  # type: ignore
                else:
                    sequence_index[kmer] = [name, prev_entry]
            else:
                sequence_index[kmer] = name
    return sequence_index


def identify_sequence(
        sequence: str,
        sequence_index: Dict[int, Union[List[str], str]],
        k: int = DEFAULT_K) -> Tuple[int, int, str]:
    kmers = canonical_kmers(sequence, k)
    counted_seqs = collections.Counter()
    for kmer in kmers:
        matched = sequence_index.get(kmer, [])
        if isinstance(matched, list):
            counted_seqs.update(matched)
        else:
            counted_seqs.update([matched])
    most_matches = 0
    best_match = "No match"
    matches = sorted(counted_seqs.items(), key=lambda tup: tup[1], reverse=True)
    if matches:
        best_match, most_matches = matches[0]
    return most_matches, len(kmers), best_match
