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

import array
import math
import os
import sys
from typing import Iterable, Iterator, List, Sequence, Tuple

import tqdm

import xopen

from ._qc import A, C, G, N, T
from ._qc import AdapterCounter, FastqParser, FastqRecordView, \
    PerTileQuality, QCMetrics, SequenceDuplication
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE
from .plots import (adapter_content_plot, base_content_plot,
                    per_base_quality_plot, per_sequence_gc_content_plot,
                    per_sequence_quality_scores_plot,
                    per_tile_graph,
                    sequence_length_distribution_plot)
from .stats import (base_content, base_weighted_categories, equidistant_ranges,
                    adapter_counts, per_base_qualities, mean_qualities,
                    aggregate_sequence_lengths, min_length, q20_bases,
                    total_gc_fraction, aggregate_count_matrix,
                    stringify_ranges, normalized_per_tile_averages, )

__all__ = [
    "A", "C", "G", "N", "T",
    "AdapterCounter",
    "FastqParser",
    "FastqRecordView",
    "PerTileQuality",
    "QCMetrics",
    "SequenceDuplication",
    "NUMBER_OF_NUCS",
    "NUMBER_OF_PHREDS",
    "PHRED_MAX",
    "TABLE_SIZE"
]


def html_report(self):
    return f"""
    <html>
    <head>
        <meta http-equiv="content-type" content="text/html:charset=utf-8">
        <title>sequali report</title>
    </head>
    <h1>sequali report</h1>
    <h2>Summary</h2>
    <table>
    <tr><td>Mean length</td><td align="right">
        {self.mean_length():.2f}</td></tr>
    <tr><td>Length range (min-max)</td><td align="right">
        {self.min_length()}-{self.max_length}</td></tr>
    <tr><td>total reads</td><td align="right">{self.total_reads}</td></tr>
    <tr><td>total bases</td><td align="right">{self.total_bases}</td></tr>
    <tr>
        <td>Q20 bases</td>
        <td align="right">
            {self.q20_bases()} ({self.q20_bases() * 100 / self.total_bases:.2f}%)
        </td>
    </tr>
    <tr>
        <td>Q28 bases</td>
        <td align="right">
            {self.q28_bases()} ({self.q28_bases() * 100 / self.total_bases:.2f}%)
        </td>
    </tr>
    <tr><td>GC content</td><td align="right">
        {self.total_gc_fraction() * 100:.2f}%
    </td></tr>
    </table>
    <h2>Quality scores</h2>
    {per_base_quality_plot(self.per_base_qualities(),
                           self.mean_qualities(),
                           self.data_categories)}
    </html>
    <h2>Sequence length distribution</h2>
    {sequence_length_distribution_plot(
        sequence_lengths=self.sequence_lengths(),
        x_labels=["0"] + self.data_categories)}
    <h2>Base content</h2>
    {base_content_plot(self.base_content(), self.data_categories)}
    <h2>Per sequence GC content</h2>
    {per_sequence_gc_content_plot(self.gc_content)}
    <h2>Per sequence quality scores</h2>
    {per_sequence_quality_scores_plot(self.phred_scores)}
    <h2>Adapter content plot</h2>
    {adapter_content_plot(self.all_adapter_values(),
                          self.adapter_counter.adapters,
                          self.data_categories)}
    </html>
    """


def main():
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
    filename = sys.argv[1]
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
    report = QCMetricsReport(metrics, adapter_counter)
    print(report.html_report())
    print(
        per_tile_graph(
            normalized_per_tile_averages(per_tile_quality),
            [f"{x}-{y}" for x, y in
             equidistant_ranges(per_tile_quality.max_length, 50)]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
