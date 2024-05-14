import json
import sys

from sequali.sequence_identification import identify_sequence_builtin


if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, "rb") as f:
        data = json.load(f)
    sequence_dicts = data["overrepresented_sequences"]["overrepresented_sequences"]
    for seqdict in sequence_dicts:
        identify_sequence_builtin(seqdict["sequence"])

