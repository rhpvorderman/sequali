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
        sequence_lookup: Dict[str, str],
        k: int = DEFAULT_K,
        match_reverse_complement: bool = True,
) -> Tuple[int, int, str]:
    kmers = canonical_kmers(sequence, k)
    counted_seqs: typing.Counter[str] = collections.Counter()
    sequence_reverse_complement = reverse_complement(sequence)
    for kmer in kmers:
        matched = sequence_index.get(kmer, [])
        if isinstance(matched, list):
            counted_seqs.update(matched)
        else:
            counted_seqs.update([matched])
    best_identity = 0.0
    best_match = "No match"
    matches = sorted(counted_seqs.items(), key=lambda tup: tup[1], reverse=True)
    for match, _ in matches:
        target_sequence = sequence_lookup[match]
        identity = sequence_identity(target_sequence, sequence)
        if match_reverse_complement:
            reverse_identity = sequence_identity(target_sequence,
                                                 sequence_reverse_complement)
            identity = max(identity, reverse_identity)
        if identity > best_identity:
            best_identity = identity
            best_match = match
            if identity == 1.0:
                break
    return round(best_identity * len(sequence)), len(sequence), best_match


def identify_sequence_builtin(sequence: str, k: int = DEFAULT_K,
                              match_reverse_complement: bool = True):
    """
    Identify a sequence using the builtin sequence libraries.
    :return: A tuple of kmer matches, the max matches and a string containing
             the best match.
    """
    sequence_index = create_default_sequence_index(k)
    sequence_lookup = dict(default_contaminant_iterator())
    return identify_sequence(sequence, sequence_index, sequence_lookup, k,
                             match_reverse_complement)


class Entry(typing.NamedTuple):
    score: int
    query_matches: int


def sequence_identity(target: str, query: str,
                      match_score=1, mismatch_penalty=-1, deletion_penalty=-1,
                      insertion_penalty=-1):
    """
    Calculate sequence identity based on a smith-waterman matrix. Only keep
    two columns in memory as no walk-back is needed.
    Identity is given as (query_length - errors / query_length).
    """
    # Store (character, score, errors) tuples.
    prev_column: List[Entry] = [Entry(0, 0) for _ in range(len(query) + 1)]
    highest_score = 0
    most_matches = 0
    for i, target_char in enumerate(target):
        new_column = prev_column[:]
        for j, query_char in enumerate(query, start=1):
            # linear
            prev_entry = prev_column[j - 1]
            if target_char == query_char:
                linear_score = prev_entry.score + match_score
                linear_matches = prev_entry.query_matches + 1
            else:
                linear_score = prev_entry.score + mismatch_penalty
                # Do not deduct matches for a mismatch
                linear_matches = prev_entry.query_matches
            # insertion
            prev_ins_entry = prev_column[j]
            prev_del_entry = new_column[j - 1]
            insertion_score = prev_ins_entry.score + insertion_penalty
            deletion_score = prev_del_entry.score + deletion_penalty

            if linear_score >= insertion_score and linear_score >= deletion_score:
                score = linear_score
                matches = linear_matches
            elif insertion_score >= deletion_score:
                score = insertion_score
                # When an insertion happens in the query in theory we can
                # match all query characters still. So deduct one as a penalty.
                matches = prev_ins_entry.query_matches - 1
            else:
                score = deletion_score
                # When a deletion happens in the query, that character cannot
                # match anything anymore. So no need to deduct a penalty.
                matches = prev_del_entry.query_matches
            if score < 0:
                score = 0
                matches = 0
            new_column[j] = Entry(score, matches)
            if score == highest_score and matches > most_matches:
                most_matches = matches
            elif score > highest_score:
                highest_score = score
                most_matches = matches
        prev_column = new_column
    return most_matches / len(query)
