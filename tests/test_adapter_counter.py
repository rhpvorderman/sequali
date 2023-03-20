import pytest

from fasterqc import AdapterCounter


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
        AdapterCounter(["A" * 31, "A" * 64])
    error.match("64")
    error.match("63")
