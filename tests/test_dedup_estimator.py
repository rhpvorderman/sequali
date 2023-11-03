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
