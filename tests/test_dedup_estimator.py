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

import itertools
import string

from sequali._qc import DedupEstimator


def test_dedup_estimator():
    dedup_est = DedupEstimator(hash_table_size_bits=8)
    assert dedup_est._hash_table_size == 1 << 8
    dedup_est.add_sequence("test")
    dedup_est.add_sequence("test2")
    dedup_est.add_sequence("test3")
    dedup_est.add_sequence("test4")
    for i in range(100):
        dedup_est.add_sequence("test5")
    dupcounts = list(dedup_est.duplication_counts())
    assert len(dupcounts) == dedup_est.tracked_sequences
    dupcounts.sort()
    assert dupcounts[-1] == 100
    assert dupcounts[0] == 1


def test_dedup_estimator_switches_modulo():
    dedup_est = DedupEstimator(8)
    assert dedup_est._modulo_bits == 1
    ten_alphabets = [string.ascii_letters] * 10
    infinite_seqs = ("".join(letters) for letters in itertools.product(*ten_alphabets))
    for i, seq in zip(range(10000), infinite_seqs):
        dedup_est.add_sequence(seq)
    assert dedup_est._modulo_bits != 1
    # 2 ** 8 * 7 // 10 = 179 seqs can be stored.
    # 10_000 / 179 = 56 sequences per slot. That requires 6 bits modulo,
    # selecting one in 64 sequences.
    assert dedup_est._modulo_bits == 6
