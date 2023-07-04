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

import tqdm

import xopen

from ._qc import AdapterCounter, FastqParser, PerTileQuality, QCMetrics, \
    SequenceDuplication
from .html_report import html_report
from .stats import calculate_stats


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input FASTQ file")
    parser.add_argument("--json",
                        help="JSON output file. default '<input>.json'")
    parser.add_argument("--html",
                        help="HTML output file. default '<input>.html'")
    parser.add_argument("--dir", help="Output directory",
                        default=os.getcwd())
    return parser


def main():
    args = argument_parser().parse_args()
    metrics = QCMetrics()
    adapters = {
        "Illumina Universal Adapter": "AGATCGGAAGAG",
        "Illumina Small RNA 3' Adapter": "TGGAATTCTCGG",
        "Illumina Small RNA 5' Adapter": "GATCGTCGGACT",
        "Nextera Transposase Sequence": "CTGTCTCTTATA",
        "PolyA": "AAAAAAAAAAAA",
        "PolyG": "GGGGGGGGGGGG",
    }
    adapter_counter = AdapterCounter(adapters.values())
    per_tile_quality = PerTileQuality()
    sequence_duplication = SequenceDuplication()
    progress_update_every_xth_byte = 1024 * 1024 * 10
    progress_update_at = progress_update_every_xth_byte
    progress_bytes = 0
    filename = args.input
    with xopen.xopen(filename, "rb", threads=0) as file:  # type: ignore
        total = os.stat(filename).st_size
        if hasattr(file, "fileobj") and file.fileobj.seekable():
            # True for gzip objects
            get_current_pos = file.fileobj.tell
        elif file.seekable():
            get_current_pos = file.tell
        else:
            total = None

            def get_current_pos():
                return progress_bytes
        progress = tqdm.tqdm(
            desc=f"Processing {os.path.basename(filename)}",
            unit="iB", unit_scale=True, unit_divisor=1024,
            total=total
        )
        with progress:
            total_bytes = 0
            reader = FastqParser(file)
            for record_array in reader:
                progress_bytes += len(record_array.obj)
                if progress_bytes > progress_update_at:
                    current_pos = get_current_pos()
                    progress.update(current_pos - total_bytes)
                    total_bytes = current_pos
                    progress_update_at += progress_update_every_xth_byte
                metrics.add_record_array(record_array)
                per_tile_quality.add_record_array(record_array)
                adapter_counter.add_record_array(record_array)
                sequence_duplication.add_record_array(record_array)
            progress.update(get_current_pos() - total_bytes)
    json_data = calculate_stats(metrics,
                                adapter_counter,
                                per_tile_quality,
                                sequence_duplication)
    if args.json is None:
        args.json = os.path.basename(filename) + ".json"
    if args.html is None:
        args.html = os.path.basename(filename) + ".html"
    with open(os.path.join(args.dir, args.json), "wt") as json_file:
        json.dump(json_data, json_file, indent=2)
    with open(os.path.join(args.dir, args.html), "wt") as html_file:
        html_file.write(html_report(json_data))


if __name__ == "__main__":  # pragma: no cover
    main()
