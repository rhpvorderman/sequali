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
