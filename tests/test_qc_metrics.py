import math

from sequali import A, C, G, N, T
from sequali import FastqRecordView, QCMetrics
from sequali import NUMBER_OF_NUCS, NUMBER_OF_PHREDS


def view_from_sequence(sequence: str) -> FastqRecordView:
    return FastqRecordView(
        "name",
        sequence,
        "A" * len(sequence)
    )


def test_qc_metrics():
    sequence = "A" * 10 + "C" * 10 + "G" * 10 + "T" * 10 + "N" * 10
    qualities = chr(10 + 33) * 25 + chr(30 + 33) * 25
    metrics = QCMetrics()
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.max_length == len(sequence)
    assert metrics.number_of_reads == 1
    gc_content = metrics.gc_content()
    assert sum(gc_content) == 1
    assert gc_content[50] == 1
    phred_content = metrics.phred_scores()
    this_read_error = (10 ** -1) * 25 + (10 ** -3) * 25
    this_read_phred = -10 * math.log10(this_read_error / len(sequence))
    phred_index = round(this_read_phred)
    assert phred_content[phred_index] == 1
    assert sum(phred_content) == 1
    phred_array = metrics.phred_count_table()
    assert len(phred_array) == len(sequence) * NUMBER_OF_PHREDS
    assert sum(phred_array[(10 // 4):len(phred_array):NUMBER_OF_PHREDS]) == 25
    assert sum(phred_array[(30 // 4):len(phred_array):NUMBER_OF_PHREDS]) == 25
    assert sum(phred_array) == len(sequence)
    for i in range(25):
        assert phred_array[(10 // 4) + NUMBER_OF_PHREDS * i] == 1
    for i in range(25, 50):
        assert phred_array[(30 // 4) + NUMBER_OF_PHREDS * i] == 1
    base_array = metrics.base_count_table()
    assert len(base_array) == len(sequence) * NUMBER_OF_NUCS
    assert sum(phred_array[A: len(phred_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array[C: len(phred_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array[G: len(phred_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array[T: len(phred_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array[N: len(phred_array): NUMBER_OF_NUCS]) == 10
    assert sum(phred_array) == len(sequence)
    for i in range(10):
        assert base_array[A + NUMBER_OF_NUCS * i] == 1
    for i in range(10, 20):
        assert base_array[C + NUMBER_OF_NUCS * i] == 1
    for i in range(20, 30):
        assert base_array[G + NUMBER_OF_NUCS * i] == 1
    for i in range(30, 40):
        assert base_array[T + NUMBER_OF_NUCS * i] == 1
    for i in range(40, 50):
        assert base_array[N + NUMBER_OF_NUCS * i] == 1


def test_long_sequence():
    metrics = QCMetrics()
    # This will test the base counting in vectors properly as that is limited
    # at 255 * 16 nucleotides (4080 bytes) and thus needs to flush the counts
    # properly.
    sequence = 4096 * 'A' + 4096 * 'C'
    qualities = 8192 * chr(20 + 33)
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.phred_scores()[20] == 1
    assert metrics.gc_content()[50] == 1


def test_average_long_quality():
    metrics = QCMetrics()
    sequence = 20_000_000 * "A"
    # After creating a big error float of 1000.0 we start adding very small
    # probabilities of 1 in 100_000. If the counting is too imprecise, the
    # rounding errors will lead to an incorrect average score for
    # the entire read.
    qualities = 1000 * chr(0 + 33) + 19_999_000 * chr(50 + 33)
    error_rate = (1 + 19999 * 10 ** -5) / 20000
    phred = -10 * math.log10(error_rate)
    metrics.add_read(FastqRecordView("name", sequence, qualities))
    assert metrics.phred_scores()[round(phred)] == 1
