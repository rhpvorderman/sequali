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

import os
import sys
from typing import Any, Dict

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
                    stringify_ranges, normalized_per_tile_averages,
                    sequence_lengths)

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


def calculate_stats(
        metrics: QCMetrics,
        adapter_counter: AdapterCounter,
        per_tile_quality: PerTileQuality,
        sequence_duplication: SequenceDuplication) -> Dict[str, Any]:
    count_table = metrics.count_table()

    data_ranges = list(equidistant_ranges(metrics.max_length, 50))
    aggregated_table = aggregate_count_matrix(count_table, data_ranges)
    total_bases = sum(aggregated_table)
    total_reads = metrics.number_of_reads
    seq_lengths = sequence_lengths(count_table, total_reads)
    x_labels = stringify_ranges(data_ranges)
    adapter_counts = adapter_counter.get_counts()
    adapter_fractions = [[
        (adapter, sum(count_array[start:stop]) / adapter_counter.number_of_sequences)
        for start, stop in data_ranges
    ] for adapter, count_array in adapter_counts]
    pbq = per_base_qualities(aggregated_table)
    return {
        "summary": {
            "mean_length":  total_bases / total_reads,
            "minimum_length": min_length(seq_lengths),
            "max_length": metrics.max_length,
            "total_reads": total_reads,
            "total_bases": total_bases,
            "q20_bases": q20_bases(aggregated_table),
            "total_gc_fraction": total_gc_fraction(aggregated_table),
        },
        "per_base_qualities": {
            "x_labels": x_labels,
            "values": {
                "mean": mean_qualities(aggregated_table),
                "A": pbq[A],
                "C": pbq[C],
                "G": pbq[G],
                "T": pbq[T],
            },
        },
        "sequence_length_distribution": {
            "x_labels": [0] + x_labels,
            "values": sequence_lengths(aggregated_table, total_reads)
        },
        "base_content": {
            "x_labels": x_labels,
            "values": base_content(aggregated_table),
        },
        "per_sequence_gc_content": {
            "x_labels": [str(i) for i in range(101)],
            "values": metrics.gc_content(),
        },
        "per_sequence_quality_scores": {
            "x_labels": [str(i) for i in range(PHRED_MAX + 1)],
            "values": metrics.phred_scores(),
        },
        "adapter_content": {
            "x_labels": x_labels,
            "values": list(zip(adapter_counter.adapters,
                                 adapter_fractions))

        },
        "per_tile_quality": {
            "skipped_reason": per_tile_quality.skipped_reason,
            "normalized_per_tile_averages": normalized_per_tile_averages(
                per_tile_quality.get_tile_averages(), data_ranges),
            "x_labels": x_labels,
        }
    }

def html_report(data: Dict[str, Any]):
    summary = data["summary"]
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
        {summary["mean_length"]:.2f}</td></tr>
    <tr><td>Length range (min-max)</td><td align="right">
        {summary["minimum_length"]}-{summary["maximum_length"]}</td></tr>
    <tr><td>total reads</td><td align="right">{summary["total_reads"]}</td></tr>
    <tr><td>total bases</td><td align="right">{summary["total_bases"]}</td></tr>
    <tr>
        <td>Q20 bases</td>
        <td align="right">
            {summary["q20_bases"]} ({summary["q20_bases"] * 100 / 
                                     summary["total_bases"]:.2f}%)
        </td>
    </tr>
    <tr><td>GC content</td><td align="right">
        {summary["total_gc_fraction"] * 100:.2f}%
    </td></tr>
    </table>
    <h2>Quality scores</h2>
    {per_base_quality_plot(data["per_base_qualities"]["values"],
                           data["per_base_qualities"]["x_labels"],)}
    </html>
    <h2>Sequence length distribution</h2>
    {sequence_length_distribution_plot(
        data["sequence_length_distribution"]["values"],
        data["sequence_length_distribution"]["x_labels"],
    )}
    <h2>Base content</h2>
    {base_content_plot(data["base_content"]["values"],
                       data["base_content"]["x_labeles"])}
    <h2>Per sequence GC content</h2>
    {per_sequence_gc_content_plot(data["per_sequence_gc_content"]["values"])}
    <h2>Per sequence quality scores</h2>
    {per_sequence_quality_scores_plot(data["per_sequence_quality_scores"]["values"])}
    <h2>Adapter content plot</h2>
    {adapter_content_plot(data["adapter_content"]["values"],
                          data["adapter_content"]["x_labels"])}
    <h2>Per Tile Quality</h2>
    {data["per_tile_quality"]["skipped_reason"] if 
        data["per_tile_quality"]["skipped_reason"]
    else
    per_tile_graph(
        data["per_tile_quality"]["normalized_per_tile_averages"],
        data["per_tile_quality"]["x_labels"]    
    )
    }
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
    json_data = calculate_stats(metrics,
                                adapter_counter,
                                per_tile_quality,
                                sequence_duplication)
    print(html_report(json_data))


if __name__ == "__main__":  # pragma: no cover
    main()
