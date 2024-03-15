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

from ._qc import (
    AdapterCounter, BamParser, DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS,
    DEFAULT_FRAGMENT_LENGTH, DEFAULT_MAX_UNIQUE_FRAGMENTS,
    DEFAULT_UNIQUE_SAMPLE_EVERY, DedupEstimator, FastqParser, NanoStats,
    PerTileQuality, QCMetrics, SequenceDuplication
)
from ._version import __version__
from .adapters import DEFAULT_ADAPTER_FILE, adapters_from_file
from .report_modules import (calculate_stats, dict_to_report_modules,
                             report_modules_to_dict, write_html_report)
from .util import (ProgressUpdater, guess_sequencing_technology_from_bam_header,
                   guess_sequencing_technology_from_file)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a quality metrics report for sequencing data.")
    parser.add_argument("input", metavar="INPUT",
                        help="Input FASTQ or uBAM file. "
                             "The format is autodetected and compressed "
                             "formats are supported.")
    parser.add_argument("--json",
                        help="JSON output file. default: '<input>.json'.")
    parser.add_argument("--html",
                        help="HTML output file. default: '<input>.html'.")
    parser.add_argument("--outdir", "--dir", metavar="OUTDIR",
                        help="Output directory for the report files. default: "
                             "current working directory.",
                        default=os.getcwd())
    parser.add_argument("--adapter-file",
                        default=DEFAULT_ADAPTER_FILE,
                        help=f"File with adapters to search for. See default "
                             f"file for formatting. "
                             f"Default: {DEFAULT_ADAPTER_FILE}.")
    parser.add_argument("--overrepresentation-threshold-fraction",
                        metavar="FRACTION",
                        type=float,
                        default=0.001,
                        help="At what fraction a sequence is determined to be "
                             "overrepresented. The threshold is calculated as "
                             "fraction times the number of sampled sequences. "
                             "Default: 0.001 (1 in 1,000)."
                        )
    parser.add_argument("--overrepresentation-min-threshold", type=int,
                        metavar="THRESHOLD",
                        default=100,
                        help=f"The minimum amount of occurrences for a sequence "
                             f"to be considered overrepresented, regardless of "
                             f"the bound set by the threshold fraction. Useful for "
                             f"smaller files. Default: {100}.")
    parser.add_argument("--overrepresentation-max-threshold", type=int,
                        metavar="THRESHOLD",
                        default=sys.maxsize,
                        help="The amount of occurrences for a sequence to be "
                             "considered overrepresented, regardless of the "
                             "bound set by the threshold fraction. Useful for "
                             "very large files. Default: unlimited.")
    parser.add_argument("--overrepresentation-max-unique-fragments",
                        type=int,
                        metavar="N",
                        default=DEFAULT_MAX_UNIQUE_FRAGMENTS,
                        help=f"The maximum amount of unique fragments to "
                             f"store. Larger amounts increase the sensitivity "
                             f"of finding overrepresented sequences at the "
                             f"cost of increasing memory usage. Default: "
                             f"{DEFAULT_MAX_UNIQUE_FRAGMENTS:,}.")
    parser.add_argument("--overrepresentation-fragment-length", type=int,
                        metavar="LENGTH",
                        default=DEFAULT_FRAGMENT_LENGTH,
                        help=f"The length of the fragments to sample. The "
                             f"maximum is 31. Default: {DEFAULT_FRAGMENT_LENGTH}.")
    parser.add_argument("--overrepresentation-sample-every", type=int,
                        default=DEFAULT_UNIQUE_SAMPLE_EVERY,
                        metavar="DIVISOR",
                        help=f"How often a read should be sampled. "
                             f"More samples leads to better precision, "
                             f"lower speed, and also towards more bias towards "
                             f"the beginning of the file as the fragment store "
                             f"gets filled up with more sequences from the "
                             f"beginning. "
                             f"Default: 1 in {DEFAULT_UNIQUE_SAMPLE_EVERY}.")
    parser.add_argument("--deduplication-estimate-bits", type=int,
                        default=DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS,
                        metavar="BITS",
                        help=f"Determines how many sequences are maximally "
                             f"stored to estimate the deduplication rate. "
                             f"Maximum stored sequences: 2 ** bits * 7 // 10. "
                             f"Memory required: 2 ** bits * 24. "
                             f"Default: {DEFAULT_DEDUP_HASH_TABLE_SIZE_BITS}.")
    parser.add_argument("-t", "--threads", type=int, default=2,
                        help="Number of threads to use. If greater than one "
                             "sequali will use an additional thread for gzip "
                             "decompression. Default: 2.")
    parser.add_argument("--version", action="version",
                        version=__version__)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    fraction_threshold = args.overrepresentation_threshold_fraction
    max_threshold = args.overrepresentation_max_threshold
    # if max_threshold is set it needs to be lower than min threshold
    min_threshold = min(args.overrepresentation_min_threshold, max_threshold)

    metrics = QCMetrics()
    per_tile_quality = PerTileQuality()
    sequence_duplication = SequenceDuplication(
        max_unique_fragments=args.overrepresentation_max_unique_fragments,
        fragment_length=args.overrepresentation_fragment_length,
        sample_every=args.overrepresentation_sample_every
    )
    dedup_estimator = DedupEstimator(
        hash_table_size_bits=args.deduplication_estimate_bits)
    nanostats = NanoStats()
    filename: str = args.input
    threads = args.threads
    if threads < 1:
        raise ValueError(f"Threads must be greater than 1, got {threads}.")
    with xopen.xopen(filename, "rb", threads=threads-1) as file:  # type: ignore
        progress = ProgressUpdater(filename, file)
        if filename.endswith(".bam") or (
                hasattr(file, "peek") and file.peek(4)[:4] == b"BAM\1"):
            reader = BamParser(file)
            seqtech = guess_sequencing_technology_from_bam_header(reader.header)
        else:
            reader = FastqParser(file)  # type: ignore
            seqtech = guess_sequencing_technology_from_file(file)  # type: ignore
        adapters = list(adapters_from_file(args.adapter_file, seqtech))
        adapter_counter = AdapterCounter(adapter.sequence for adapter in adapters)
        with progress:
            for record_array in reader:
                metrics.add_record_array(record_array)
                per_tile_quality.add_record_array(record_array)
                adapter_counter.add_record_array(record_array)
                sequence_duplication.add_record_array(record_array)
                nanostats.add_record_array(record_array)
                dedup_estimator.add_record_array(record_array)
                progress.update(record_array)
    report_modules = calculate_stats(
        filename,
        metrics,
        adapter_counter,
        per_tile_quality,
        sequence_duplication,
        dedup_estimator,
        nanostats,
        adapters=adapters,
        fraction_threshold=fraction_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold)
    if args.json is None:
        args.json = os.path.basename(filename) + ".json"
    if args.html is None:
        args.html = os.path.basename(filename) + ".html"
    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.isabs(args.json):
        args.json = os.path.join(args.outdir, args.json)
    if not os.path.isabs(args.html):
        args.html = os.path.join(args.outdir, args.html)
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
    timestamp = os.stat(in_json).st_mtime
    write_html_report(dict_to_report_modules(json_data), output,
                      output.rstrip(".html"), timestamp)
