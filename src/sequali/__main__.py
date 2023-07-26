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
import json
import os
import sys

import xopen

from ._qc import AdapterCounter, FastqParser, PerTileQuality, QCMetrics, \
    SequenceDuplication
from .html_report import html_report
from .stats import calculate_stats
from .util import ProgressUpdater, sequence_file_iterator

DEFAULT_ADAPTERS_FILE = os.path.join(os.path.dirname(__file__),
                                     "adapters", "adapter_list.txt")


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

    return parser


def main():
    args = argument_parser().parse_args()
    fraction_threshold = args.overrepresentation_threshold_fraction
    max_threshold = args.overrepresentation_max_threshold
    # if max_threshold is set it needs to be lower than min threshold
    min_threshold = min(args.overrepresentation_min_threshold, max_threshold)

    metrics = QCMetrics()
    adapters = dict(sequence_file_iterator(DEFAULT_ADAPTERS_FILE))
    adapter_counter = AdapterCounter(adapters.values())
    per_tile_quality = PerTileQuality()
    sequence_duplication = SequenceDuplication()
    filename = args.input
    with xopen.xopen(filename, "rb", threads=0) as file:  # type: ignore
        progress = ProgressUpdater(filename, file)
        with progress:
            reader = FastqParser(file)
            for record_array in reader:
                metrics.add_record_array(record_array)
                per_tile_quality.add_record_array(record_array)
                adapter_counter.add_record_array(record_array)
                sequence_duplication.add_record_array(record_array)
                progress.update(record_array)
    json_data = calculate_stats(metrics,
                                adapter_counter,
                                per_tile_quality,
                                sequence_duplication,
                                adapter_names=list(adapters.keys()),
                                fraction_threshold=fraction_threshold,
                                min_threshold=min_threshold,
                                max_threshold=max_threshold)
    if args.json is None:
        args.json = os.path.basename(filename) + ".json"
    if args.html is None:
        args.html = os.path.basename(filename) + ".html"
    if not os.path.isabs(args.json):
        args.json = os.path.join(args.dir, args.json)
    if not os.path.isabs(args.html):
        args.html = os.path.join(args.dir, args.html)
    with open(args.json, "wt") as json_file:
        # Indent=0 is ~40% smaller than indent=2 while still human-readable
        json.dump(json_data, json_file, indent=0)
    with open(args.html, "wt") as html_file:
        html_file.write(html_report(json_data))


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
    with open(output, "wt") as html:
        html.write(html_report(json_data))
