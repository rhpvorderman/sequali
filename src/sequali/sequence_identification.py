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
import collections
import functools
import os
import typing
from typing import Dict, Iterable, Iterator, List, Tuple, Union

from .util import fasta_parser

DEFAULT_K = 13

CONTAMINANTS_DIR = os.path.join(os.path.dirname(__file__), "contaminants")
DEFAULT_CONTAMINANTS_FILES = [f.path for f in os.scandir(CONTAMINANTS_DIR)
                              if f.name != "README"]


def default_contaminant_iterator() -> Iterator[Tuple[str, str]]:
    for file in DEFAULT_CONTAMINANTS_FILES:
        yield from fasta_parser(file)


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
    canonical_set = set()
    # Do all the translation upfront and not for each kmer individually as
    # there will be a lot of overlapping work
    # Encoding to bytes makes translating faster
    seq_bytes = sequence.encode("ascii")
    upper_seq_bytes = seq_bytes.translate(UPPER_TABLE)
    revcomp_seq = upper_seq_bytes.translate(COMPLEMENT_TABLE)[::-1].decode('ascii')
    upper_seq = upper_seq_bytes.decode('ascii')
    seqlen = len(sequence)
    for i in range(seqlen + 1 - k):
        kmer = upper_seq[i:i+k]
        revcomp_end = seqlen - i
        revcomp = revcomp_seq[revcomp_end - k:revcomp_end]
        if revcomp < kmer:
            canonical_set.add(revcomp)
        else:
            canonical_set.add(kmer)
    return canonical_set


def create_sequence_index(
        names_and_sequences: Iterable[Tuple[str, str]],
        k: int = DEFAULT_K,
        ) -> Dict[str, Union[List[str], str]]:
    sequence_index: Dict[str, Union[List[str], str]] = {}
    for name, sequence in names_and_sequences:
        kmers = canonical_kmers(sequence, k)
        # Store one sequence name (most common) or alternatively a list
        # of multiple names. This is a bit convoluted, but it
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


@functools.lru_cache
def create_default_sequence_index(k: int = DEFAULT_K
                                  ) -> Dict[str, Union[List[str], str]]:
    return create_sequence_index(default_contaminant_iterator(), k)


def identify_sequence(
        sequence: str,
        sequence_index: Dict[str, Union[List[str], str]],
        k: int = DEFAULT_K) -> Tuple[int, int, str]:
    kmers = canonical_kmers(sequence, k)
    counted_seqs: typing.Counter[str] = collections.Counter()
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


def identify_sequence_builtin(sequence: str, k: int = DEFAULT_K):
    """
    Identify a sequence using the builtin sequence libraries.
    :return: A tuple of kmer matches, the max matches and a string containing
             the best match.
    """
    sequence_index = create_default_sequence_index(k)
    return identify_sequence(sequence, sequence_index, k)
