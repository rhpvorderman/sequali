# Copyright (C) 2023 Leiden University Medical Center
# This file is part of Sequali
#
# Sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Sequali.  If not, see <https://www.gnu.org/licenses/

import argparse
import contextlib
import json
import os
import sys


from ._qc import (
    AdapterCounter,
    DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS,
    DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH,
    DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET,
    DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH,
    DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET,
    DEFAULT_FRAGMENT_LENGTH,
    DEFAULT_MAX_UNIQUE_FRAGMENTS,
    DEFAULT_UNIQUE_SAMPLE_EVERY,
    DedupEstimator,
    InsertSizeMetrics,
    NanoStats,
    PerTileQuality,
    QCMetrics,
    SequenceDuplication
)
from ._version import __version__
from .adapters import DEFAULT_ADAPTER_FILE, adapters_from_file
from .report_modules import (calculate_stats, dict_to_report_modules,
                             report_modules_to_dict, write_html_report)
from .util import NGSFile, sequence_names_match

DEFAULT_FINGERPRINT_BACK_SEQUENCE_PAIRED_OFFSET = 0
DEFAULT_FINGERPRINT_FRONT_SEQUENCE_PAIRED_OFFSET = 0


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a quality metrics report for sequencing data.")
    parser.add_argument("input", metavar="INPUT",
                        help="Input FASTQ or uBAM file. "
                             "The format is autodetected and compressed "
                             "formats are supported.")
    parser.add_argument("input_reverse", metavar="INPUT_REVERSE",
                        nargs="?",
                        help="Second FASTQ file for Illumina paired-end reads."
                        )
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
                        help="The minimum amount of occurrences for a sequence "
                             "to be considered overrepresented, regardless of "
                             "the bound set by the threshold fraction. Useful for "
                             "smaller files. Default: 100.")
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
    parser.add_argument("--duplication-max-stored-fingerprints", type=int,
                        default=DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS,
                        metavar="N",
                        help=f"Determines how many fingerprints are maximally "
                             f"stored to estimate the duplication rate. "
                             f"More fingerprints leads to a more accurate "
                             f"estimate, but also more memory usage. "
                             f"Default: {DEFAULT_DEDUP_MAX_STORED_FINGERPRINTS:,}.")
    parser.add_argument("--fingerprint-front-length", type=int,
                        default=DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH,
                        metavar="LENGTH",
                        help=f"Set the number of bases to be taken for the "
                             f"deduplication fingerprint from the front of "
                             f"the sequence. "
                             f"Default: {DEFAULT_FINGERPRINT_FRONT_SEQUENCE_LENGTH}.")

    parser.add_argument("--fingerprint-back-length", type=int,
                        default=DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH,
                        metavar="LENGTH",
                        help=f"Set the number of bases to be taken for the "
                             f"deduplication fingerprint from the back of "
                             f"the sequence. "
                             f"Default: {DEFAULT_FINGERPRINT_BACK_SEQUENCE_LENGTH}.")
    parser.add_argument("--fingerprint-front-offset", type=int,
                        metavar="LENGTH",
                        help=f"Set the offset for the front part of the "
                             f"deduplication fingerprint. Useful for avoiding "
                             f"adapter sequences. "
                             f"Default: {DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET} "
                             f"for single end, "
                             f"{DEFAULT_FINGERPRINT_FRONT_SEQUENCE_PAIRED_OFFSET} "
                             f"for paired sequences.")
    parser.add_argument("--fingerprint-back-offset", type=int,
                        metavar="LENGTH",
                        help=f"Set the offset for the back part of the "
                             f"deduplication fingerprint. Useful for avoiding "
                             f"adapter sequences. "
                             f"Default: {DEFAULT_FINGERPRINT_BACK_SEQUENCE_OFFSET} "
                             f"for single end, "
                             f"{DEFAULT_FINGERPRINT_BACK_SEQUENCE_PAIRED_OFFSET} "
                             f"for paired sequences.")
    parser.add_argument("-t", "--threads", type=int, default=2,
                        help="Number of threads to use. If greater than one "
                             "an additional thread for gzip "
                             "decompression will be used. Default: 2.")
    parser.add_argument("--version", action="version",
                        version=__version__)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    threads = args.threads
    if threads < 1:
        raise ValueError(f"Threads must be greater than 1, got {threads}.")

    fraction_threshold = args.overrepresentation_threshold_fraction
    max_threshold = args.overrepresentation_max_threshold
    # if max_threshold is set it needs to be lower than min threshold
    min_threshold = min(args.overrepresentation_min_threshold, max_threshold)
    paired = bool(args.input_reverse)

    metrics1 = QCMetrics()
    per_tile_quality1 = PerTileQuality()
    nanostats1 = NanoStats()
    sequence_duplication1 = SequenceDuplication(
        max_unique_fragments=args.overrepresentation_max_unique_fragments,
        fragment_length=args.overrepresentation_fragment_length,
        sample_every=args.overrepresentation_sample_every
    )
    if paired:
        if args.fingerprint_front_offset is None:
            args.fingerprint_front_offset = (
                DEFAULT_FINGERPRINT_FRONT_SEQUENCE_PAIRED_OFFSET)
        if args.fingerprint_back_offset is None:
            args.fingerprint_back_offset = (
                DEFAULT_FINGERPRINT_BACK_SEQUENCE_PAIRED_OFFSET)
    else:
        if args.fingerprint_front_offset is None:
            args.fingerprint_front_offset = (
                DEFAULT_FINGERPRINT_FRONT_SEQUENCE_OFFSET)
        if args.fingerprint_back_offset is None:
            args.fingerprint_back_offset = (
                DEFAULT_FINGERPRINT_BACK_SEQUENCE_PAIRED_OFFSET)
    dedup_estimator = DedupEstimator(
        max_stored_fingerprints=args.duplication_max_stored_fingerprints,
        front_sequence_length=args.fingerprint_front_length,
        front_sequence_offset=args.fingerprint_front_offset,
        back_sequence_length=args.fingerprint_back_length,
        back_sequence_offset=args.fingerprint_back_offset,
    )

    if paired:
        insert_size_metrics = InsertSizeMetrics()
        metrics2 = QCMetrics()
        per_tile_quality2 = PerTileQuality()
        sequence_duplication2 = SequenceDuplication(
            max_unique_fragments=args.overrepresentation_max_unique_fragments,
            fragment_length=args.overrepresentation_fragment_length,
            sample_every=args.overrepresentation_sample_every
        )
    else:
        metrics2 = None
        per_tile_quality2 = None
        sequence_duplication2 = None
        insert_size_metrics = None
    with contextlib.ExitStack() as exit_stack:
        reader1 = NGSFile(args.input, threads - 1)
        exit_stack.enter_context(reader1)
        seqtech = reader1.sequencing_technology
        if paired:
            reader2 = NGSFile(args.input_reverse, threads - 1)
            exit_stack.enter_context(reader2)
            if reader1.sequencing_technology != reader2.sequencing_technology:
                raise RuntimeError(
                    f"Mismatching sequencing technologies:\n"
                    f"{reader1.filepath}: {reader1.sequencing_technology}\n"
                    f"{reader2.filepath}: {reader2.sequencing_technology}\n")
            if not (reader1.format == "FASTQ" and reader2.format == "FASTQ"):
                raise RuntimeError("Paired end mode is only supported for "
                                   "FASTQ files.")
            seqtech = "illumina"  # Paired end is always illumina
            adapter_counter1 = None
        adapters = list(adapters_from_file(args.adapter_file, seqtech))
        if not paired:
            adapter_counter1 = AdapterCounter(
                adapter.sequence for adapter in adapters)
        for record_array1 in reader1:
            metrics1.add_record_array(record_array1)
            per_tile_quality1.add_record_array(record_array1)
            sequence_duplication1.add_record_array(record_array1)
            nanostats1.add_record_array(record_array1)
            if paired:
                record_array2 = reader2.read(len(record_array1))
                if len(record_array1) != len(record_array2):
                    raise RuntimeError(
                        f"FASTQ Files out of sync {args.input} has more "
                        f"FASTQ records than {args.input_reverse}.")
                if not (record_array1.is_mate(record_array2)):
                    for r1, r2 in zip(iter(record_array1),
                                      iter(record_array2)):
                        if not sequence_names_match(r1.name(),
                                                    r2.name()):
                            raise RuntimeError(
                                f"Mismatching names found! {r1.name()} "
                                f"{r2.name()}")
                    raise RuntimeError("Mismatching names found!")
                dedup_estimator.add_record_array_pair(record_array1, record_array2)
                insert_size_metrics.add_record_array_pair(record_array1, record_array2)  # type: ignore  # noqa: E501
                metrics2.add_record_array(record_array2)  # type: ignore  # noqa: E501
                per_tile_quality2.add_record_array(record_array2)  # type: ignore  # noqa: E501
                sequence_duplication2.add_record_array(record_array2)  # type: ignore  # noqa: E501
            else:
                adapter_counter1.add_record_array(record_array1)   # type: ignore  # noqa: E501
                dedup_estimator.add_record_array(record_array1)
        if paired and len(reader2.read(1)) > 0:
            raise RuntimeError(
                f"FASTQ Files out of sync {args.input_reverse} has "
                f"more FASTQ records than {args.input}.")
    report_modules = calculate_stats(
        filename=args.input,
        metrics=metrics1,
        adapter_counter=adapter_counter1,
        per_tile_quality=per_tile_quality1,
        sequence_duplication=sequence_duplication1,
        dedup_estimator=dedup_estimator,
        nanostats=nanostats1,
        insert_size_metrics=insert_size_metrics,
        filename_reverse=args.input_reverse,
        metrics_reverse=metrics2,
        per_tile_quality_reverse=per_tile_quality2,
        sequence_duplication_reverse=sequence_duplication2,
        adapters=adapters,
        fraction_threshold=fraction_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold)
    os.makedirs(args.outdir, exist_ok=True)
    if args.json is None:
        args.json = os.path.basename(args.input) + ".json"
    if args.html is None:
        args.html = os.path.basename(args.input) + ".html"
    if not os.path.isabs(args.json):
        args.json = os.path.join(args.outdir, args.json)
    if not os.path.isabs(args.html):
        args.html = os.path.join(args.outdir, args.html)
    with open(args.json, "wt") as json_file:
        json_dict = report_modules_to_dict(report_modules)
        # Indent=0 is ~40% smaller than indent=2 while still human-readable
        json.dump(json_dict, json_file, indent=0)
    write_html_report(report_modules, args.html)


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
    write_html_report(dict_to_report_modules(json_data), output)
