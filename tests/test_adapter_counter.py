from fasterqc import AdapterCounter
from fasterqc._qc import MAX_SEQUENCE_SIZE

import pytest


def test_adapter_counter_basic_init():
    adapters = [
        "IF I SHOULD STAY",
        "I WILL ONLY BE IN YOUR WAY",
        "SO I'LL GO, BUT I KNOW",
        "I'LL THINK OF YOU EVERY STEP OF THE WAY",
        "AND I WILL ALWAYS LOVE YOU"
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
        AdapterCounter(["correct", b'incorrect'])
    error.match("b'incorrect'")


def test_adapter_count_init_non_ascii():
    with pytest.raises(ValueError) as error:
        AdapterCounter(["correct", "íncørrect"])
    error.match("ASCII")
    error.match("'íncørrect'")


def test_adapter_count_init_too_long():
    with pytest.raises(ValueError) as error:
        AdapterCounter(["A" * 31, "A" * (MAX_SEQUENCE_SIZE + 1)])
    error.match(str(MAX_SEQUENCE_SIZE + 1))
    error.match(str(MAX_SEQUENCE_SIZE))


def test_adapter_counter_matcher():
    counter = AdapterCounter(["singing", "rain", "train"])
    # Only first match should be counted.
    sequence = ("I'm singing in the rain, just singing in the rain\n"
                "What a glorious feeling, I'm happy again")
    counter.add_sequence(sequence)
    counts = counter.get_counts()
    assert counts[0][0] == "singing"
    assert counts[1][0] == "rain"
    assert counts[2][0] == "train"
    singing_list = counts[0][1].tolist()
    assert len(singing_list) == len(sequence)
    assert singing_list[sequence.find("singing")] == 1
    rain_list = counts[1][1].tolist()
    assert len(rain_list) == len(sequence)
    assert rain_list[sequence.find("rain")] == 1
    assert sum(singing_list) == 1
    assert sum(rain_list) == 1
    train_list = counts[2][1].tolist()
    assert sum(train_list) == 0


def test_adapter_counter_add_sequence_no_string():
    counter = AdapterCounter(["GATTACA"])
    with pytest.raises(TypeError) as error:
        counter.add_sequence(b"GATATATACCACA")  # type: ignore
    error.match("str")
    error.match("bytes")


def test_adapter_counter_add_sequence_no_ascii():
    counter = AdapterCounter(["GATTACA"])
    with pytest.raises(ValueError) as error:
        counter.add_sequence("GÅTTAÇA")
    error.match("GÅTTAÇA")
    error.match("ASCII")


def test_adapter_counter_matcher_multiple_machine_words():
    adapters = [
        "A" * MAX_SEQUENCE_SIZE,
        "C" * MAX_SEQUENCE_SIZE,
        "G" * MAX_SEQUENCE_SIZE,
        "T" * MAX_SEQUENCE_SIZE,
    ]
    sequence = ("GATTACA" * 20).join(adapters)
    counter = AdapterCounter(adapters)
    counter.add_sequence(sequence)
    for adapter, countview in counter.get_counts():
        assert countview[sequence.find(adapter)] == 1
        assert sum(countview) == 1
