# Copyright (C) 2023 Leiden University Medical Center
# This file is part of sequali
#
# sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with sequali.  If not, see <https://www.gnu.org/licenses/
import argparse
import io
import json
import os
import sys

import xopen

from ._qc import AdapterCounter, DEFAULT_MAX_UNIQUE_SEQUENCES, FastqParser, \
    NanoStats, PerTileQuality, QCMetrics, SequenceDuplication
from .adapters import DEFAULT_ADAPTER_FILE, adapters_from_file
from .report_modules import (calculate_stats, dict_to_report_modules,
                             report_modules_to_dict, write_html_report)
from .util import ProgressUpdater, guess_sequencing_technology


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input FASTQ file")
    parser.add_argument("--json",
                        help="JSON output file. default: '<input>.json'")
    parser.add_argument("--html",
                        help="HTML output file. default: '<input>.html'")
    parser.add_argument("--dir", help="Output directory. default: "
                                      "current working directory",
                        default=os.getcwd())
    parser.add_argument("--overrepresentation-threshold-fraction", type=float,
                        default=0.0001,
                        help="At what fraction a sequence is determined to be "
                             "overrepresented. Default: 0.0001 (1 in 100 000)."
                        )
    parser.add_argument("--overrepresentation-min-threshold", type=int,
                        default=100,
                        help="The minimum amount of sequences that need to be "
                             "present to be considered overrepresented even if "
                             "the threshold fraction is surpassed. Useful for "
                             "smaller files.")
    parser.add_argument("--overrepresentation-max-threshold", type=int,
                        default=sys.maxsize,
                        help="The threshold above which a sequence is "
                             "considered overrepresented even if the "
                             "threshold fraction is not surpassed. Useful for "
                             "very large files.")
    parser.add_argument("--max-unique-sequences", type=int,
                        default=DEFAULT_MAX_UNIQUE_SEQUENCES,
                        help="The maximum amount of unique sequences to "
                             "gather. Larger amounts increase the sensitivity "
                             "of finding overrepresented sequences and "
                             "increase the accuracy of the duplication "
                             "estimate, at the cost of increasing memory "
                             "usage at about 50 bytes per sequence.")

    return parser


def main():
    args = argument_parser().parse_args()
    fraction_threshold = args.overrepresentation_threshold_fraction
    max_threshold = args.overrepresentation_max_threshold
    # if max_threshold is set it needs to be lower than min threshold
    min_threshold = min(args.overrepresentation_min_threshold, max_threshold)

    metrics = QCMetrics()
    per_tile_quality = PerTileQuality()
    sequence_duplication = SequenceDuplication(args.max_unique_sequences)
    nanostats = NanoStats()
    filename = args.input
    with xopen.xopen(filename, "rb", threads=0) as file:  # type: ignore
        progress = ProgressUpdater(filename, file)
        try:
            # Guess sequencing technology to limit the amount of adapter probes
            # needed to search.
            seqtech = guess_sequencing_technology(file.peek(io.DEFAULT_BUFFER_SIZE))
        except IOError:
            seqtech = None
        adapters = list(adapters_from_file(DEFAULT_ADAPTER_FILE, seqtech))
        adapter_counter = AdapterCounter(adapter.sequence for adapter in adapters)
        with progress:
            reader = FastqParser(file)
            for record_array in reader:
                metrics.add_record_array(record_array)
                per_tile_quality.add_record_array(record_array)
                adapter_counter.add_record_array(record_array)
                sequence_duplication.add_record_array(record_array)
                nanostats.add_record_array(record_array)
                progress.update(record_array)
    report_modules = calculate_stats(
        metrics,
        adapter_counter,
        per_tile_quality,
        sequence_duplication,
        nanostats,
        adapter_names=list(adapter.name for adapter in adapters),
        fraction_threshold=fraction_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold)
    if args.json is None:
        args.json = os.path.basename(filename) + ".json"
    if args.html is None:
        args.html = os.path.basename(filename) + ".html"
    os.makedirs(args.dir, exist_ok=True)
    if not os.path.isabs(args.json):
        args.json = os.path.join(args.dir, args.json)
    if not os.path.isabs(args.html):
        args.html = os.path.join(args.dir, args.html)
    with open(args.json, "wt") as json_file:
        json_dict = report_modules_to_dict(report_modules)
        # Indent=0 is ~40% smaller than indent=2 while still human-readable
        json.dump(json_dict, json_file, indent=0)
    write_html_report(report_modules, args.html, filename)


if __name__ == "__main__":  # pragma: no cover
    main()


def sequali_report():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", metavar="JSON", help="Sequali JSON file")
    parser.add_argument("-o", "--html", help="Output html file default: "
                                             "<input>.html")
    args = parser.parse_args()
    output = args.html
    in_json = args.json
    if not output:
        # Remove json extension and add HTML
        output = ".".join(in_json.split(".")[:-1]) + ".html"
    with open(in_json) as j:
        json_data = json.load(j)
    write_html_report(dict_to_report_modules(json_data), output,
                      output.rstrip(".html"))
