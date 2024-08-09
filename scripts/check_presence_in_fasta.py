import argparse

from sequali.util import fasta_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta")
    parser.add_argument("sequence")
    args = parser.parse_args()

    total = 0
    sequence = args.sequence.upper()
    for name, contig in fasta_parser(args.fasta):
        contig = contig.upper()
        total += contig.count(sequence)
    print(total)


if __name__ == "__main__":
    main()
