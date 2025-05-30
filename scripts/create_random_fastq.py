import dnaio
import array
import random


def create_random_fastqs():
    # Reproducible random. That is:
    random.seed(0)
    for length in range(501):
        tile = random.randrange(0, 1000)
        x = random.randrange(0, 100_000)
        y = random.randrange(0, 100_000)
        name = f"SIM:1234:ABC123:1:{tile}:{x}:{y}"
        sequence = "".join(random.choices("ACGT", k=length))
        qs = array.array("B", random.choices(range(33, 127), k=length))
        qualities = qs.tobytes().decode("ascii")
        yield name, sequence, qualities


with dnaio.open("tests/data/random_seqs.fastq", mode="w") as fastq:
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    for name, sequence, qualities in create_random_fastqs():
        seq = dnaio.SequenceRecord(
            name=name,
            sequence=sequence,
            qualities=qualities,
        )
        fastq.write(seq)
