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

import pytest

from sequali.sequence_identification import canonical_kmers, reverse_complement


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
