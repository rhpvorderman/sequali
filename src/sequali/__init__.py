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

import pygal  # type: ignore

import tqdm

import xopen

from ._qc import A, C, G, N, T
from ._qc import AdapterCounter, FastqParser, FastqRecordView, \
    PerTileQuality, QCMetrics, SequenceDuplication
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE

PHRED_TO_ERROR_RATE = [
    sum(10 ** (-p / 10) for p in range(start * 4, start * 4 + 4)) / 4
    for start in range(NUMBER_OF_PHREDS)
]

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


def equidistant_ranges(length: int, parts: int) -> Iterator[Tuple[int, int]]:
    size = length // parts
    remainder = length % parts
    small_parts = parts - remainder
    start = 0
    for i in range(parts):
        part_size = size if i < small_parts else size + 1
        if part_size == 0:
            continue
        stop = start + part_size
        yield start, stop
        start = stop


def base_weighted_categories(
    base_counts: Sequence[int], number_of_categories: int
) -> Iterator[Tuple[int, int]]:
    total_bases = sum(base_counts)
    per_category = total_bases // number_of_categories
    enough_bases = per_category
    start = 0
    total = 0
    for stop, count in enumerate(base_counts, start=1):
        total += count
        if total >= enough_bases:
            yield start, stop
            start = stop
            enough_bases += per_category
    if start != len(base_counts):
        yield start, len(base_counts)


def cumulative_percentages(counts: Iterable[int], total: int):
    cumalitive_percentages = []
    count_sum = 0
    for count in counts:
        count_sum += count
        cumalitive_percentages.append(count_sum / total)
    return cumalitive_percentages


def per_tile_graph(per_tile_quality: PerTileQuality) -> str:
    tile_averages = per_tile_quality.get_tile_averages()
    max_length = per_tile_quality.max_length
    ranges = list(equidistant_ranges(max_length, 50))

    average_phreds = []
    per_category_totals = [0.0 for i in range(len(ranges))]
    for tile, averages in tile_averages:
        range_averages = [sum(averages[start:stop]) / (stop - start)
                          for start, stop in ranges]
        range_phreds = []
        for i, average in enumerate(range_averages):
            phred = -10 * math.log10(average)
            range_phreds.append(phred)
            # Averaging phreds takes geometric mean.
            per_category_totals[i] += phred
        average_phreds.append((tile, range_phreds))
    number_of_tiles = len(tile_averages)
    averages_per_category = [total / number_of_tiles
                             for total in per_category_totals]
    scatter_plot = pygal.Line(
        title="Sequence length distribution",
        x_labels=[f"{start}-{stop}" for start, stop in ranges],
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
        stroke=False,
    )

    for tile, tile_phreds in average_phreds:
        normalized_tile_phreds = [
            tile_phred - average
            for tile_phred, average in zip(tile_phreds, averages_per_category)
        ]
        scatter_plot.add(str(tile),  normalized_tile_phreds)

    return scatter_plot.render(is_unicode=True)


def per_base_quality_plot(per_base_qualities: Sequence[Sequence[float]],
                          mean_qualities: Sequence[float],
                          data_categories: Sequence[str]) -> str:
    plot = pygal.Line(
        title="Per base sequence quality",
        dots_size=1,
        x_labels=data_categories,
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    plot.add("A", per_base_qualities[A])
    plot.add("C", per_base_qualities[C])
    plot.add("G", per_base_qualities[G])
    plot.add("T", per_base_qualities[T])
    plot.add("mean", mean_qualities)
    return plot.render(is_unicode=True)


def sequence_length_distribution_plot(sequence_lengths: Sequence[int],
                                      x_labels: Sequence[str]) -> str:
    plot = pygal.Bar(
        title="Sequence length distribution",
        x_labels=x_labels,
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    plot.add("Length", sequence_lengths)
    return plot.render(is_unicode=True)


def base_content_plot(base_content: Sequence[Sequence[float]],
                      x_labels: Sequence[str]) -> str:
    plot = pygal.StackedLine(
        title="Base content",
        dots_size=1,
        x_labels=x_labels,
        y_labels=[i / 10 for i in range(11)],
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    plot.add("G", base_content[G], fill=True)
    plot.add("C", base_content[C], fill=True)
    plot.add("A", base_content[A], fill=True)
    plot.add("T", base_content[T], fill=True)
    plot.add("N", base_content[N], fill=True)
    return plot.render(is_unicode=True)


def per_sequence_gc_content_plot(gc_content: Sequence[int]) -> str:
    plot = pygal.Bar(
        title="Per sequence GC content",
        x_labels=range(101),
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    plot.add("", gc_content)
    return plot.render(is_unicode=True)


def per_sequence_quality_scores_plot(
        per_sequence_quality_scores: Sequence[int]) -> str:
    plot = pygal.Line(
        title="Per sequence quality scores",
        x_labels=range(PHRED_MAX + 1),
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    total = sum(per_sequence_quality_scores)
    percentage_scores = [100 * score / total
                         for score in per_sequence_quality_scores]
    plot.add("%", percentage_scores)
    return plot.render(is_unicode=True)


def adapter_content_plot(adapter_content: Sequence[Sequence[float]],
                         adapter_labels: Sequence[str],
                         x_labels: Sequence[str],) -> str:
    plot = pygal.Line(
        title="Adapter content (%)",
        x_labels=x_labels,
        range=(0.0, 100.0),
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    for label, content in zip(adapter_labels, adapter_content):
        plot.add(label, content)
    return plot.render(is_unicode=True)


class QCMetricsReport:
    raw_count_matrix: array.ArrayType
    aggregated_count_matrix: array.ArrayType
    raw_sequence_lengths: array.ArrayType
    gc_content: array.ArrayType
    phred_scores: array.ArrayType
    _data_ranges: List[Tuple[int, int]]
    data_categories: List[str]
    max_length: int
    total_reads: int
    total_bases: int

    def __init__(self, metrics: QCMetrics, adapter_counter: AdapterCounter,
                 graph_resolution: int = 100):
        """Aggregate all data from a QCMetrics counter"""

        self.adapter_counter = adapter_counter
        self.total_reads = metrics.number_of_reads
        assert metrics.number_of_reads == adapter_counter.number_of_sequences
        self.max_length = metrics.max_length
        self.raw_count_matrix = array.array("Q")
        # Python will treat the memoryview as an iterable in the array constructor
        # use from_bytes instead for direct memcpy.
        self.raw_count_matrix = metrics.count_table()
        self.gc_content = metrics.gc_content()
        self.phred_scores = metrics.phred_scores()

        matrix = memoryview(self.raw_count_matrix)

        # use bytes constructor to initialize the aggregated count matrix to 0.
        raw_sequence_lengths = array.array("Q", bytes(8 * (self.max_length + 1)))
        raw_base_counts = array.array("Q", bytes(8 * (self.max_length + 1)))
        # All reads have at least 0 bases
        raw_base_counts[0] = self.total_reads
        for i in range(self.max_length):
            table = matrix[i * 60:(i + 1) * 60]
            raw_base_counts[i + 1] = sum(table)

        previous_count = 0
        for i in range(self.max_length, 0, -1):
            number_at_least = raw_base_counts[i]
            raw_sequence_lengths[i] = number_at_least - previous_count
            previous_count = number_at_least
        self.raw_sequence_lengths = raw_sequence_lengths
        self.total_bases = sum(memoryview(raw_base_counts)[1:])

        expected_bases_per_position = self.total_bases / self.max_length
        if raw_base_counts[-1] > (0.5 * expected_bases_per_position):
            # Most reads are max length. Use an equidistant distribution
            self._data_ranges = list(
                equidistant_ranges(metrics.max_length, graph_resolution)
            )
        else:
            # We have greater variance in the length of the reads. Use the
            # base counts to get roughly equal base counts per category.
            self._data_ranges = list(
                base_weighted_categories(
                    memoryview(raw_base_counts)[1:], graph_resolution
                )
            )

        # Use one-based indexing for the graph categories. I.e. 1 is the first base.
        self.data_categories = [
            f"{start + 1}-{stop}" if start + 1 != stop else f"{start + 1}"
            for start, stop in self._data_ranges
        ]

        self.aggregated_count_matrix = array.array(
            "Q", bytes(8 * TABLE_SIZE * len(self.data_categories))
        )
        categories_view = memoryview(self.aggregated_count_matrix)
        table_size = TABLE_SIZE
        for cat_index, (start, stop) in enumerate(self._data_ranges):
            cat_offset = cat_index * table_size
            cat_view = categories_view[cat_offset: cat_offset + table_size]
            for table_index in range(start, stop):
                offset = table_index * table_size
                table = matrix[offset: offset + table_size]
                for i, count in enumerate(table):
                    cat_view[i] += count

    def _tables(self) -> Iterator[memoryview]:
        category_view = memoryview(self.aggregated_count_matrix)
        for i in range(0, len(category_view), TABLE_SIZE):
            yield category_view[i: i + TABLE_SIZE]

    def base_content(self) -> List[List[float]]:
        content = [
            [0.0 for _ in range(len(self.data_categories))]
            for _ in range(NUMBER_OF_NUCS)
        ]
        for cat_index, table in enumerate(self._tables()):
            total = sum(table)
            if total == 0:
                continue
            for i in range(NUMBER_OF_NUCS):
                content[i][cat_index] = sum(table[i::NUMBER_OF_NUCS]) / total
        return content

    def total_gc_fraction(self) -> float:
        total_nucs = [
            sum(
                self.aggregated_count_matrix[
                    i: len(self.aggregated_count_matrix): NUMBER_OF_NUCS
                ]
            )
            for i in range(NUMBER_OF_NUCS)
        ]
        at = total_nucs[A] + total_nucs[T]
        gc = total_nucs[G] + total_nucs[C]
        return gc / (at + gc)

    def q20_bases(self):
        q20s = 0
        for table in self._tables():
            q20s += sum(table[NUMBER_OF_NUCS * 5:])
        return q20s

    def q28_bases(self):
        q28s = 0
        for table in self._tables():
            q28s += sum(table[NUMBER_OF_NUCS * 7:])
        return q28s

    def min_length(self) -> int:
        for length, count in enumerate(self.raw_sequence_lengths):
            if count > 0:
                return length
        return 0

    def mean_length(self) -> float:
        return self.total_bases / self.total_reads

    def sequence_lengths(self):
        seqlength_view = memoryview(self.raw_sequence_lengths)[1:]
        lengths = [sum(seqlength_view[start:stop]) for start, stop in self._data_ranges]
        return [self.raw_sequence_lengths[0]] + lengths

    def mean_qualities(self):
        mean_qualites = [0.0 for _ in range(len(self.data_categories))]
        for cat_index, table in enumerate(self._tables()):
            total = 0
            total_prob = 0.0
            for phred_p_value, offset in zip(
                PHRED_TO_ERROR_RATE, range(0, TABLE_SIZE, NUMBER_OF_NUCS)
            ):
                nucs = table[offset: offset + NUMBER_OF_NUCS]
                count = sum(nucs)
                total += count
                total_prob += count * phred_p_value
            if total == 0:
                continue
            mean_qualites[cat_index] = -10 * math.log10(total_prob / total)
        return mean_qualites

    def per_base_qualities(self) -> List[List[float]]:
        base_qualities = [
            [0.0 for _ in range(len(self.data_categories))]
            for _ in range(NUMBER_OF_NUCS)
        ]
        for cat_index, table in enumerate(self._tables()):
            nuc_probs = [0.0 for _ in range(NUMBER_OF_NUCS)]
            nuc_counts = [0 for _ in range(NUMBER_OF_NUCS)]
            for phred_p_value, offset in zip(
                PHRED_TO_ERROR_RATE, range(0, TABLE_SIZE, NUMBER_OF_NUCS)
            ):
                nucs = table[offset: offset + NUMBER_OF_NUCS]
                for i, count in enumerate(nucs):
                    nuc_counts[i] += count
                    nuc_probs[i] += phred_p_value * count
            for i in range(NUMBER_OF_NUCS):
                if nuc_counts[i] == 0:
                    continue
                base_qualities[i][cat_index] = -10 * math.log10(
                    nuc_probs[i] / nuc_counts[i]
                )
        return base_qualities

    def all_adapter_values(self):
        all_adapters = []
        for adapter, countarray in self.adapter_counter.get_counts():
            adapter_counts = [sum(countarray[start:stop])
                              for start, stop in self._data_ranges]
            total = 0
            accumulated_counts = []
            for count in adapter_counts:
                total += count
                accumulated_counts.append(total)
            all_adapters.append([count * 100 / self.total_reads for
                                 count in accumulated_counts])
        return all_adapters

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
    print(per_tile_graph(per_tile_quality))


if __name__ == "__main__":  # pragma: no cover
    main()
