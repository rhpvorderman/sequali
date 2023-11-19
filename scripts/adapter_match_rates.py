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
    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("allowed_errors", type=int)
    args = parser.parse_args()
    length = args.length
    allowed_errors = args.allowed_errors
    print(f"Length: {length}. Allowed errors: {allowed_errors}.")
    error_rates = (0.001, 0.01, 0.1, 0.2, 0.4, 0.5)
    print("\nChance of matching")
    print("Error rate:     \t" + "\t".join(f"{e:.4f}" for e in error_rates))
    print("Chance of match:\t" + "\t".join(
        f"{p:.4f}"
        for p in match_probabilities(error_rates, args.length, args.allowed_errors)))
    lengths = (151, 8000, 20_000)
    print("\nFalse positive rates")
    print("Length:              \t" + "\t".join(str(l) for l in lengths))
    print("False positive rate:\t" + "\t".join(
        f"{false_positive_rate(args.length, l, args.allowed_errors):.5f}"
        for l in lengths))


if __name__ == "__main__":
    main()

