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

from sequali import AdapterCounter
from sequali._qc import FastqRecordView, MAX_SEQUENCE_SIZE


def test_adapter_counter_basic_init():
    adapters = [
        "GATTTAGAGACATA",
        "TATACCCGTACCACAGAT",
        "GCCCGGGAAATTAGGCACGATT",
        "GCAGAGAGATATAGAGATACACACAGAGAGAGAT",
        "GGGCACCACAGAGACCACACAGAGACA"
    ]
    counter = AdapterCounter(adapters)
    assert counter.adapters == tuple(adapters)
    assert counter.max_length == 0
    assert counter.number_of_sequences == 0


def test_adapter_counter_init_no_iterable():
    with pytest.raises(TypeError) as error:
        AdapterCounter(1)  # type: ignore
    error.match("not iterable")


def test_adapter_count_init_no_adapters():
    with pytest.raises(ValueError) as error:
        AdapterCounter([])
    error.match("t least one")


def test_adapter_count_init_contains_non_str():
    with pytest.raises(TypeError) as error:
        AdapterCounter(["GATTACA", b'GATTACA'])
    error.match("b'GATTACA'")


def test_adapter_count_init_non_ascii():
    with pytest.raises(ValueError) as error:
        AdapterCounter(["GATTACA", "Gättaca"])
    error.match("ASCII")
    error.match("'Gättaca'")


def test_adapter_count_init_too_long():
    with pytest.raises(ValueError) as error:
        AdapterCounter(["A" * 31, "A" * (MAX_SEQUENCE_SIZE + 1)])
    error.match(str(MAX_SEQUENCE_SIZE + 1))
    error.match(str(MAX_SEQUENCE_SIZE))


def test_adapter_counter_matcher():
    counter = AdapterCounter(["GATTACA", "GGGG", "TTTTT"])
    # Only first match should be counted.
    sequence = ("AAGATTACAAAAAGATTACAGGGGAACGAGGGG")
    read = FastqRecordView("bla", sequence, "H" * len(sequence))
    counter.add_read(read)
    counts = counter.get_counts()
    assert counts[0][0] == "GATTACA"
    assert counts[1][0] == "GGGG"
    assert counts[2][0] == "TTTTT"
    gattaca_list = counts[0][1].tolist()
    assert len(gattaca_list) == len(sequence)
    assert gattaca_list[sequence.find("GATTACA")] == 1
    gggg_list = counts[1][1].tolist()
    assert len(gggg_list) == len(sequence)
    assert gggg_list[sequence.find("GGGG")] == 1
    assert sum(gattaca_list) == 1
    assert sum(gggg_list) == 1
    ttttt_list = counts[2][1].tolist()
    assert sum(ttttt_list) == 0


def test_adapter_counter_add_read_no_view():
    counter = AdapterCounter(["GATTACA"])
    with pytest.raises(TypeError) as error:
        counter.add_read(b"GATATATACCACA")  # type: ignore
    error.match("FastqRecordView")
    error.match("bytes")


@pytest.mark.parametrize("adapters", [
    [  # Creates 2 SSE vectors
        "A" * MAX_SEQUENCE_SIZE,
        "C" * MAX_SEQUENCE_SIZE,
        "G" * MAX_SEQUENCE_SIZE,
        "T" * MAX_SEQUENCE_SIZE,
    ],
    [  # Creates 1 SSE Vector and one 64-bit integer.
        "A" * MAX_SEQUENCE_SIZE,
        "C" * MAX_SEQUENCE_SIZE,
        "G" * MAX_SEQUENCE_SIZE,
    ],
    [  # Creates 1 SSE Vector.
        "A" * MAX_SEQUENCE_SIZE,
        "C" * MAX_SEQUENCE_SIZE,
    ],
    [  # Creates 2 SSE Vectors and one 64-bit integer.
        "A" * MAX_SEQUENCE_SIZE,
        "C" * MAX_SEQUENCE_SIZE,
        "G" * MAX_SEQUENCE_SIZE,
        "T" * MAX_SEQUENCE_SIZE,
        "N" * MAX_SEQUENCE_SIZE,
    ],
])
def test_adapter_counter_matcher_multiple_machine_words(adapters):
    sequence = ("GATTACA" * 20).join(adapters)
    read = FastqRecordView("name", sequence, "H" * len(sequence))
    counter = AdapterCounter(adapters)
    counter.add_read(read)
    for adapter, forward_counts, reverse_counts in counter.get_counts():
        index = sequence.find(adapter)
        reverse_index = len(sequence) - 1 - index
        assert forward_counts[index] == 1
        assert reverse_counts[reverse_index] == 1
        assert sum(forward_counts) == 1
        assert sum(reverse_counts) == 1


def test_adapter_counter_mixed_lengths():
    adapters = [
        "TATAAATATAAATATAAA",
        "GATTACAGATTACAGATTACA",
        "AAAAAAAAAAAA",
        "TTTTTTTTTTTT",
        "CACGTCAGTTACCGGATAGA",
        "GGTCAAGGGGTAAATGATAT",
        "AGGTAGATTTATTTTATTTAT",
        "GGGTGGGAGGCC"
    ]
    sequences = [
        "NNNNN".join(adapters[i] for i in range(8)),
        "NNNNNNN".join(adapters[i] for i in (1, 2, 4, 5, 6, 7)),
        "NNN".join(adapters[i] for i in (7, 2, 6, 3, 4, 7)),
    ]
    counter = AdapterCounter(adapters)
    for sequence in sequences:
        read = FastqRecordView("name", sequence, "H" * len(sequence))
        counter.add_read(read)
    for sequence in sequences:
        for adapter, forward_counts, reverse_counts in counter.get_counts():
            forward_counts = forward_counts.tolist()
            index = sequence.find(adapter)

            if index != -1:
                reverse_index = len(sequence) - 1 - index
                assert sequence[index] == sequence[::-1][reverse_index]
                assert forward_counts[index] > 0
                assert reverse_counts[reverse_index] > 0
