import dnaio


with dnaio.open("tests/data/100_illumina_adapters.fastq", mode="w") as fastq:
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    for i in range(100):
        seq = dnaio.SequenceRecord(
            name=f"SIM:1234:ABC123:1234:1234:{i * 100}:{i * 25} 1:N:0:12",
            sequence=adapter,
            qualities=chr(30 + 33) * len(adapter),
        )
        fastq.write(seq)
