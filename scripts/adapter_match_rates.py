"""
Calculate adapter match rates.

For adapter probe size selection two metrics are important:
  - The chance of a random hit on a random dna sequence of a given length.
  - The chance the probe will be detected if it is there for a given error rate.

This script calculates both allowing to make some design choices
"""
import argparse
import math


def binominal(rate, length, events):
    """
    Calulate the probability of a given number of events occuring given the
    rate and an interval.

    See: https://en.wikipedia.org/wiki/Poisson_distribution
    """
    return math.comb(length, events) * rate ** events * (1 - rate) ** (length - events)


def chance_to_match(length, error_rate, allowed_errors):
    """
    Calculate the chance a probe of length matches a sequence with the
    error_rate and up to allowed errors.
    """
    if allowed_errors < 0:
        raise ValueError(
            f"allowed_errors should be a positive number, got {allowed_errors}")
    return math.fsum(
        binominal(error_rate, length, n) for n in range(allowed_errors + 1)
    )


def match_probabilities(error_rates, length, allowed_errors):
    return [chance_to_match(length, error_rate, allowed_errors)
            for error_rate in error_rates]


def chance_accidental_match(length, allowed_errors):
    return math.fsum(
        math.comb(length, i) / 4 ** (length - i)
        for i in range(allowed_errors + 1)
    )


def false_positive_rate(probe_length, sequence_length, allowed_errors):
    change_no_false_positive = 1 - chance_accidental_match(probe_length, allowed_errors)
    return 1 - change_no_false_positive ** sequence_length


def main():
    lengths_and_errors = [
        (12, 0),
        (12, 1),
        (16, 0),
        (16, 1),
        (21, 1),
        (21, 2),
        (21, 3),
        (21, 4),
        (32, 4),
        (32, 6),
        (32, 8),
        (32, 10),
        (64, 10),
        (64, 16),
        (64, 20),
        (64, 22),
        (64, 23),
        (63, 24),
    ]
    error_rates = (0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5)
    lengths = (151, 8000, 20_000)
    print("probe\t\tProbe detection rate for amount of sequencing errors\tFalse positive rates for read length")
    print("length\terrors\t" + "\t".join(f"{e:>7.2%}" for e in error_rates) +
          "\t" + "\t".join(f"{l:>7,}" for l in lengths))

    for probe_length, allowed_errors in lengths_and_errors:
        detection_rates = "\t".join(
            f"{p:>7.2%}"
            for p in
            match_probabilities(error_rates, probe_length, allowed_errors)
        )
        false_positive_rates = "\t".join(
            f"{false_positive_rate(probe_length, l, allowed_errors):>7.3%}"
            for l in lengths
        )
        print(f"{probe_length}\t{allowed_errors}\t{detection_rates}\t{false_positive_rates}")


if __name__ == "__main__":
    main()

