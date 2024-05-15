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

import pytest

from sequali.sequence_identification import (
    canonical_kmers,
    reverse_complement,
    sequence_identity
)


@pytest.mark.parametrize(["sequence", "rev_complement"], [
    ("ATA", "TAT"),
    ("ATCG", "CGAT"),
    ("ANTCG", "CGANT")
])
def test_reverse_complement(sequence: str, rev_complement: str):
    assert reverse_complement(sequence) == rev_complement


def test_canonical_kmers():
    assert canonical_kmers("GATTACA", 3) == {"ATC", "AAT", "TAA", "GTA", "ACA"}
    assert canonical_kmers("gattaca", 3) == \
           canonical_kmers(reverse_complement("GATTACA"), 3)


@pytest.mark.parametrize(["target", "query", "result"], [
    ("XXXACGTXXX", "ACGT", 1.0),
    ("ACGT", "ACGT", 1.0),
    ("ACXGT", "ACGT", 3 / 4),  # All characters match but subtract -1 for the insertion.
    ("ACGT", "ACXGT", 4 / 5),  # One out of 5 characters was deleted.
    ("ACGT", "AGGT", 3 / 4),
    ("XXACXGTXXXACGTXXXX", "ACGT", 1.0),  # Most optimal match should be found.
    # Example from
    # https://en.wikipedia.org/wiki/File:Smith-Waterman-Algorithm-Example-Step3.png
    # Alignment is:
    # TGTT-ACGG
    # X|||-||XX
    # GGTTGACTA
    # 5 out of 9 matches
    ("TGTTACGG", "GGTTGACTA", 5 / 9),
])
def test_sequence_identity(target, query, result):
    assert sequence_identity(target, query) == result
