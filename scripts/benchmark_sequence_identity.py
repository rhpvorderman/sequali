import json
import sys

from sequali.sequence_identification import identify_sequence_builtin


if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, "rb") as f:
        data = json.load(f)
    sequence_dicts = data["overrepresented_sequences"]["overrepresented_sequences"]
    for seqdict in sequence_dicts:
        best_match = seqdict["best_match"]
        total, max, found_match = identify_sequence_builtin(seqdict["sequence"])
        if best_match != found_match:
            raise ValueError(f"{best_match} != {found_match}")
