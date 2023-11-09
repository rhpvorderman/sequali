import array
import collections
import dataclasses
import io
import math
import os
import sys
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Set, Tuple, Type)

import pygal  # type: ignore
import pygal.style  # type: ignore

from ._qc import A, C, G, N, T
from ._qc import (AdapterCounter, DedupEstimator, NanoStats, PerTileQuality,
                  QCMetrics, SequenceDuplication)
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX
from .sequence_identification import DEFAULT_CONTAMINANTS_FILES, DEFAULT_K, \
    create_sequence_index, identify_sequence
from .util import fasta_parser

PHRED_INDEX_TO_ERROR_RATE = [
    sum(10 ** (-p / 10) for p in range(start * 4, start * 4 + 4)) / 4
    for start in range(NUMBER_OF_PHREDS)
]
PHRED_INDEX_TO_PHRED = [-10 * math.log10(PHRED_INDEX_TO_ERROR_RATE[i])
                        for i in range(NUMBER_OF_PHREDS)]

QUALITY_SERIES_NAMES = (
    "0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27", "28-31",
    "32-35", "36-39", "40-43", ">=44")

QUALITY_COLORS = dict(
    dark_red="#8B0000",  # 0-3
    red="#ff0000",  # 4-7
    light_red="#ff9999",  # 8-11
    almost_white_blue="#f5f5ff",  # 12-15
    very_light_blue="#e6e6ff",  # 16-19
    light_blue="#8080ff",  # 20-23
    blue="#0000FF",  # 24-27
    darker_blue="#0000b3",  # 28-31
    more_darker_blue="#000080",  # 32-35
    yet_more_darker_blue="#00004d",  # 36-39
    almost_black_blue="#000033",  # 40-43
    black="#000000",  # >=44
)

COLOR_GREEN = "#33cc33"
COLOR_RED = "#ff0000"

QUALITY_DISTRIBUTION_STYLE = pygal.style.Style(colors=list(QUALITY_COLORS.values()))
ONE_SERIE_STYLE = pygal.style.DefaultStyle(colors=("#33cc33",))  # Green
MULTIPLE_SERIES_STYLE = pygal.style.DefaultStyle()

DEFAULT_FRACTION_THRESHOLD = 0.0001
DEFAULT_MIN_THRESHOLD = 100
DEFAULT_MAX_THRESHOLD = sys.maxsize

COMMON_GRAPH_OPTIONS = dict(
    truncate_label=-1,
    width=1500,
    explicit_size=True,
    disable_xml_declaration=True,
    js=[],  # Script is globally downloaded once
)


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


def logarithmic_ranges(length: int, min_distance: int = 5):
    """
    Gives a squashed logarithmic range. It is not truly logarithmic as the
    minimum distance ensures that the lower units are more tightly packed.
    """
    # Use a scaling factor: this needs 400 units to reach the length of the
    # largest human chromosome. This will still fit on a graph once we reach
    # those sequencing sizes.
    scaling_factor = 250_000_000 ** (1 / 400)
    i = 0
    start = 0
    while True:
        stop = round(scaling_factor ** i)
        i += 1
        if stop >= start + min_distance:
            yield start, stop
            start = stop
            if stop >= length:
                return


def stringify_ranges(data_ranges: Iterable[Tuple[int, int]]):
    return [
        f"{start + 1}-{stop}" if start + 1 != stop else f"{start + 1}"
        for start, stop in data_ranges
    ]


def table_iterator(count_tables: array.ArrayType,
                   table_size: int) -> Iterator[memoryview]:
    table_view = memoryview(count_tables)
    for i in range(0, len(count_tables), table_size):
        yield table_view[i: i + table_size]


def aggregate_count_matrix(
        count_tables: array.ArrayType,
        data_ranges: Sequence[Tuple[int, int]],
        table_size: int) -> array.ArrayType:
    count_view = memoryview(count_tables)
    aggregated_matrix = array.array(
        "Q", bytes(8 * table_size * len(data_ranges)))
    ag_view = memoryview(aggregated_matrix)
    for cat_index, (start, stop) in enumerate(data_ranges):
        cat_offset = cat_index * table_size
        cat_view = ag_view[cat_offset:cat_offset + table_size]
        table_start = start * table_size
        table_stop = stop * table_size
        for i in range(table_size):
            cat_view[i] = sum(count_view[table_start + i: table_stop: table_size])
    return aggregated_matrix


def aggregate_base_tables(
        count_tables: array.ArrayType,
        data_ranges: Sequence[Tuple[int, int]],) -> array.ArrayType:
    return aggregate_count_matrix(count_tables, data_ranges, NUMBER_OF_NUCS)


def aggregate_phred_tables(
        count_tables: array.ArrayType,
        data_ranges: Sequence[Tuple[int, int]],) -> array.ArrayType:
    return aggregate_count_matrix(count_tables, data_ranges, NUMBER_OF_PHREDS)


def label_settings(x_labels: Sequence[str]) -> Dict[str, Any]:
    # Labels are ranges such as 1-5, 101-142 etc. This clutters the x axis
    # labeling so only use the first number. The values will be labelled
    # separately.
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    return dict(
        x_labels=simple_x_labels,
        x_labels_major_every=round(len(x_labels) / 30),
        x_label_rotation=30 if len(simple_x_labels[-1]) > 4 else 0,
        show_minor_x_labels=False
    )


def label_values(values: Sequence[Any], labels: Sequence[Any]):
    if len(values) != len(labels):
        raise ValueError("labels and values should have the same length")
    return [{"value": value, "label": label} for value, label
            in zip(values, labels)]


@dataclasses.dataclass
class ReportModule(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @abstractmethod
    def to_html(self) -> str:
        pass


@dataclasses.dataclass
class Summary(ReportModule):
    mean_length: float
    minimum_length: int
    maximum_length: int
    total_reads: int
    total_bases: int
    q20_bases: int
    total_gc_fraction: float

    def to_html(self) -> str:
        return f"""
            <h2>Summary</h2>
            <table>
            <tr><td>Mean length</td><td align="right">
                {self.mean_length:,.2f}</td></tr>
            <tr><td>Length range (min-max)</td><td align="right">
                {self.minimum_length:,} - {self.maximum_length:,}</td></tr>
            <tr><td>total reads</td><td align="right">{self.total_reads:,}</td></tr>
            <tr><td>total bases</td><td align="right">{self.total_bases:,}</td></tr>
            <tr>
                <td>Q20 bases</td>
                <td align="right">
                    {self.q20_bases:,} ({self.q20_bases * 100 / self.total_bases:.2f}%)
                </td>
            </tr>
            <tr><td>GC content</td><td align="right">
                {self.total_gc_fraction * 100:.2f}%
            </td></tr>
            </table>
        """


@dataclasses.dataclass
class SequenceLengthDistribution(ReportModule):
    length_ranges: List[str]
    counts: List[int]
    q1: int
    q5: int
    q10: int
    q25: int
    q50: int
    q75: int
    q90: int
    q95: int
    q99: int

    def plot(self) -> str:
        plot = pygal.Bar(
            title="Sequence length distribution",
            x_title="sequence length",
            y_title="number of reads",
            style=ONE_SERIE_STYLE,
            **label_settings(self.length_ranges),
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("Length", label_values(self.counts, self.length_ranges))
        return plot.render(is_unicode=True)

    def to_html(self):
        return f"""
            <h2>Sequence length distribution</h2>
            <table>
                <tr><td>N1</td><td align="right">{self.q1:,}</td></tr>
                <tr><td>N5</td><td align="right">{self.q5:,}</td></tr>
                <tr><td>N10</td><td align="right">{self.q10:,}</td></tr>
                <tr><td>N25</td><td align="right">{self.q25:,}</td></tr>
                <tr><td>N50</td><td align="right">{self.q50:,}</td></tr>
                <tr><td>N75</td><td align="right">{self.q75:,}</td></tr>
                <tr><td>N90</td><td align="right">{self.q90:,}</td></tr>
                <tr><td>N95</td><td align="right">{self.q95:,}</td></tr>
                <tr><td>N99</td><td align="right">{self.q99:,}</td></tr>
            </table>
            <figure>
            {self.plot()}
            </figure>
        """

    @classmethod
    def from_base_count_tables(cls,
                               base_count_tables: array.ArrayType,
                               total_sequences: int,
                               data_ranges: Sequence[Tuple[int, int]]):
        max_length = len(base_count_tables) // NUMBER_OF_NUCS
        # use bytes constructor to initialize to 0
        sequence_lengths = array.array("Q", bytes(8 * (max_length + 1)))
        base_counts = array.array("Q", bytes(8 * (max_length + 1)))
        base_counts[0] = total_sequences  # all reads have at least 0 bases
        for i, table in enumerate(table_iterator(base_count_tables, NUMBER_OF_NUCS)):
            base_counts[i + 1] = sum(table)
        previous_count = 0
        for i in range(max_length, 0, -1):
            number_at_least = base_counts[i]
            sequence_lengths[i] = number_at_least - previous_count
            previous_count = number_at_least
        seqlength_view = memoryview(sequence_lengths)[1:]
        lengths = [sum(seqlength_view[start:stop]) for start, stop in
                   data_ranges]
        x_labels = stringify_ranges(data_ranges)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_thresholds = [int(p * total_sequences / 100) for p in percentiles]
        thresh_iter = enumerate(percentile_thresholds)
        thresh_index, current_threshold = next(thresh_iter)
        accumulated_count = 0
        percentile_lengths = [0 for _ in percentiles]
        done = False
        for length, count in enumerate(sequence_lengths):
            while count > 0 and not done:
                remaining_threshold = current_threshold - accumulated_count
                if count > remaining_threshold:
                    accumulated_count += remaining_threshold
                    percentile_lengths[thresh_index] = length
                    count -= remaining_threshold
                    try:
                        thresh_index, current_threshold = next(thresh_iter)
                    except StopIteration:
                        done = True
                        break
                    continue
                break
            accumulated_count += count
            if done:
                break

        return cls(["0"] + x_labels, [sequence_lengths[0]] + lengths,
                   *percentile_lengths)


@dataclasses.dataclass
class PerPositionMeanQualityAndSpread(ReportModule):
    x_labels: List[str]
    percentiles: List[Tuple[str, List[float]]]

    def plot(self) -> str:
        plot = pygal.Line(
            title="Per position quality percentiles",
            show_dots=False,
            x_title="position",
            y_title="phred score",
            y_labels=list(range(0, 51, 10)),
            style=pygal.style.DefaultStyle(colors=["#000000"] * 12),
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        percentiles = dict(self.percentiles)
        plot.add("top 1%", label_values(percentiles["top 1%"], self.x_labels),
                 stroke_style={"dasharray": '1,2'})
        plot.add("top 5%", label_values(percentiles["top 5%"], self.x_labels),
                 stroke_style={"dasharray": '3,3'})
        plot.add("mean", label_values(percentiles["mean"], self.x_labels),
                 show_dots=True, dots_size=1)
        plot.add("bottom 5%", label_values(percentiles["bottom 5%"], self.x_labels),
                 stroke_style={"dasharray": '3,3'})
        plot.add("bottom 1%", label_values(percentiles["bottom 1%"], self.x_labels),
                 stroke_style={"dasharray": '1,2'})
        return plot.render(is_unicode=True)

    def to_html(self):
        return f"""
            <h2>Per position quality percentiles</h2>
            Shows the mean for all bases and the means of the lowest and
            highest percentiles to indicate the spread. Since the graph is
            based on the sampled categories, rather than exact phreds, it is
            an approximation.<br>
            {self.plot()}
        """

    @classmethod
    def from_phred_table_and_labels(cls, phred_tables: array.ArrayType, x_labels):
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_fractions = [i / 100 for i in percentiles]
        total_tables = len(phred_tables) // NUMBER_OF_PHREDS
        percentile_tables = [[0.0 for _ in range(total_tables)]
                             for _ in percentiles]
        reversed_percentile_tables = [[0.0 for _ in range(total_tables)]
                                      for _ in percentiles]
        mean = [0.0 for _ in range(total_tables)]
        for cat_index, table in enumerate(
                table_iterator(phred_tables, NUMBER_OF_PHREDS)):
            total = sum(table)
            total_error_rate = sum(
                PHRED_INDEX_TO_ERROR_RATE[i] * x for i, x in enumerate(table))
            percentile_thresholds = [int(f * total) for f in percentile_fractions]
            mean[cat_index] = -10 * math.log10(total_error_rate / total)
            accumulated_count = 0
            accumulated_errors = 0.0
            threshold_iter = enumerate(percentile_thresholds)
            thresh_index, current_threshold = next(threshold_iter)
            for phred_index, count in enumerate(table):
                while count > 0:
                    remaining_threshold = current_threshold - accumulated_count
                    if count > remaining_threshold:
                        accumulated_errors += (remaining_threshold *
                                               PHRED_INDEX_TO_ERROR_RATE[phred_index])
                        accumulated_count += remaining_threshold
                        percentile_tables[thresh_index][cat_index] = (
                            -10 * math.log10(
                                accumulated_errors / accumulated_count))
                        reversed_percentile_tables[thresh_index][cat_index] = (
                            -10 * math.log10(
                                (total_error_rate - accumulated_errors) /
                                (total - accumulated_count)
                            )
                        )
                        count -= remaining_threshold
                        try:
                            thresh_index, current_threshold = next(threshold_iter)
                        except StopIteration:
                            # This will make sure the next cat_index is reached
                            # since 2 ** 65 will not be reached
                            thresh_index = sys.maxsize
                            current_threshold = 2**65
                        continue
                    break
                accumulated_count += count
                accumulated_errors += PHRED_INDEX_TO_ERROR_RATE[phred_index] * count
        graph_series = [
            ("bottom 1%", percentile_tables[0]),
            ("bottom 5%", percentile_tables[1]),
            ("bottom 10%", percentile_tables[2]),
            ("bottom 25%", percentile_tables[3]),
            ("bottom 50%", percentile_tables[4]),
            ("mean", mean),
            ("top 50%", reversed_percentile_tables[-5]),
            ("top 25%", reversed_percentile_tables[-4]),
            ("top 10%", reversed_percentile_tables[-3]),
            ("top 5%", reversed_percentile_tables[-2]),
            ("top 1%", reversed_percentile_tables[-1]),
        ]
        return cls(
            x_labels=x_labels,
            percentiles=graph_series
            )


@dataclasses.dataclass
class PerBaseQualityScoreDistribution(ReportModule):
    x_labels: Sequence[str]
    series: Sequence[Sequence[float]]

    @classmethod
    def from_phred_count_table_and_labels(
            cls, phred_tables: array.ArrayType, x_labels: Sequence[str]):
        total_tables = len(x_labels)
        quality_distribution = [
            [0.0 for _ in range(total_tables)]
            for _ in range(NUMBER_OF_PHREDS)
        ]
        for cat_index, table in enumerate(
                table_iterator(phred_tables, NUMBER_OF_PHREDS)):
            total_nucs = sum(table)
            for offset, phred_count in enumerate(table):
                if phred_count == 0:
                    continue
                nuc_fraction = phred_count / total_nucs
                quality_distribution[offset][cat_index] = nuc_fraction
        return cls(x_labels, quality_distribution)

    def plot(self) -> str:
        plot = pygal.StackedBar(
            title="Per base quality distribution",
            style=QUALITY_DISTRIBUTION_STYLE,
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="position",
            y_title="fraction",
            fill=True,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        for name, serie in zip(QUALITY_SERIES_NAMES, self.series):
            serie_filled = sum(serie) > 0.0
            plot.add(name, label_values(serie, self.x_labels),
                     show_dots=serie_filled)
        return plot.render(is_unicode=True)

    def to_html(self):
        return f"""
            <h2>Per position quality score distribution</h2>
            {self.plot()}
        """


@dataclasses.dataclass
class PerSequenceAverageQualityScores(ReportModule):
    average_quality_counts: Sequence[int]
    x_labels: Tuple[str, ...] = tuple(str(x) for x in range(PHRED_MAX + 1))

    def plot(self) -> str:
        maximum_score = 0
        for i, count in enumerate(self.average_quality_counts):
            if count > 0:
                maximum_score = i
        maximum_score = max(maximum_score + 2, 40)
        plot = pygal.Bar(
            title="Per sequence quality scores",
            x_labels=range(maximum_score + 1),
            x_labels_major_every=3,
            show_minor_x_labels=False,
            style=ONE_SERIE_STYLE,
            x_title="Phred score",
            y_title="Percentage of total",
            **COMMON_GRAPH_OPTIONS
        )
        total = sum(self.average_quality_counts)
        percentage_scores = [100 * score / total
                             for score in self.average_quality_counts]

        plot.add("", percentage_scores[:maximum_score])
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per sequence average quality scores</h2>
            {self.plot()}
        """

    @classmethod
    def from_qc_metrics(cls, metrics: QCMetrics):
        return cls(list(metrics.phred_scores()))


@dataclasses.dataclass
class PerPositionBaseContent(ReportModule):
    x_labels: Sequence[str]
    A: Sequence[float]
    C: Sequence[float]
    G: Sequence[float]
    T: Sequence[float]

    def plot(self):
        style_class = pygal.style.Style
        red = "#DC143C"  # Crimson
        dark_red = "#8B0000"  # DarkRed
        blue = "#00BFFF"  # DeepSkyBlue
        dark_blue = "#1E90FF"  # DodgerBlue
        black = "#000000"
        style = style_class(
            colors=(red, dark_red, blue, dark_blue, black)
        )
        plot = pygal.StackedLine(
            title="Base content",
            style=style,
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="position",
            y_title="fraction",
            fill=True,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("G", label_values(self.G, self.x_labels))
        plot.add("C", label_values(self.C, self.x_labels))
        plot.add("A", label_values(self.A, self.x_labels))
        plot.add("T", label_values(self.T, self.x_labels))
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
             <h2>Per position base content</h2>
             {self.plot()}
        """

    @classmethod
    def from_base_count_tables_and_labels(cls,
                                          base_count_tables: array.ArrayType,
                                          labels: Sequence[str]):
        total_tables = len(base_count_tables) // NUMBER_OF_NUCS
        base_fractions = [
            [0.0 for _ in range(total_tables)]
            for _ in range(NUMBER_OF_NUCS)
        ]
        for index, table in enumerate(
                table_iterator(base_count_tables, NUMBER_OF_NUCS)):
            total_bases = sum(table)
            n_bases = table[N]
            named_total = total_bases - n_bases
            if named_total == 0:
                continue
            base_fractions[A][index] = table[A] / named_total
            base_fractions[C][index] = table[C] / named_total
            base_fractions[G][index] = table[G] / named_total
            base_fractions[T][index] = table[T] / named_total
        return cls(
            labels,
            A=base_fractions[A],
            C=base_fractions[C],
            G=base_fractions[G],
            T=base_fractions[T]
        )


@dataclasses.dataclass
class PerPositionNContent(ReportModule):
    x_labels: Sequence[str]
    n_content: Sequence[float]

    @classmethod
    def from_base_count_tables_and_labels(
            cls, base_count_tables: array.ArrayType, labels: Sequence[str]):
        total_tables = len(base_count_tables) // NUMBER_OF_NUCS
        n_fractions = [0.0 for _ in range(total_tables)]
        for index, table in enumerate(
                table_iterator(base_count_tables, NUMBER_OF_NUCS)):
            total_bases = sum(table)
            if total_bases == 0:
                continue
            n_fractions[index] = table[N] / total_bases
        return cls(
            labels,
            n_fractions
        )

    def plot(self):
        plot = pygal.Bar(
            title="Per position N content",
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="position",
            y_title="fraction",
            fill=True,
            style=ONE_SERIE_STYLE,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("N", label_values(self.n_content, self.x_labels))
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per position N content</h2>
            {self.plot()}
        """


@dataclasses.dataclass
class PerSequenceGCContent(ReportModule):
    gc_content_counts: Sequence[int]
    smoothened_gc_content_counts: Sequence[int]
    x_labels: Sequence[str] = tuple(str(x) for x in range(101))
    smoothened_x_labels: Sequence[str] = tuple(str(x) for x in range(0, 101, 2))

    def plot(self):
        plot = pygal.Bar(
            title="Per sequence GC content",
            x_labels=self.x_labels,
            x_labels_major_every=3,
            show_minor_x_labels=False,
            x_title="GC %",
            y_title="number of reads",
            style=ONE_SERIE_STYLE,
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("", self.gc_content_counts)
        return plot.render(is_unicode=True)

    def smoothened_plot(self):
        plot = pygal.Line(
            title="Per sequence GC content (smoothened)",
            x_labels=self.smoothened_x_labels,
            x_labels_major_every=3,
            show_minor_x_labels=False,
            x_title="GC %",
            y_title="number of reads",
            interpolate="cubic",
            style=ONE_SERIE_STYLE,
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("", self.smoothened_gc_content_counts)
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per sequence GC content</h2>
            For short reads with fixed size (i.e. Illumina) the plot will
            look very spiky due to the GC content calculation: GC bases / all
            bases. For read lengths of 151, both 75 and 76 GC bases lead to a
            percentage of 50% (rounded) and 72 and 73 GC bases leads to 48%
            (rounded). Only 74 GC bases leads to 49%. As a result the
            even categories will be twice as high, which creates a spike. The
            smoothened plot is provided to give a clearer picture in this case.
            <br>
            {self.plot()}
            {self.smoothened_plot()}
        """

    @classmethod
    def from_qc_metrics(cls, metrics: QCMetrics):
        gc_content = list(metrics.gc_content())
        smoothened_gc_content = []
        gc_content_iter = iter(gc_content)
        for i in range(50):
            smoothened_gc_content.append(next(gc_content_iter) + next(gc_content_iter))
        # Append the last 100% category.
        smoothened_gc_content.append(next(gc_content_iter))
        return cls(gc_content, smoothened_gc_content)


@dataclasses.dataclass
class AdapterContent(ReportModule):
    x_labels: Sequence[str]
    adapter_content: Sequence[Tuple[str, Sequence[float]]]

    def plot(self):
        plot = pygal.Line(
            title="Adapter content (%)",
            range=(0.0, 100.0),
            x_title="position",
            y_title="%",
            legend_at_bottom=True,
            legend_at_bottom_columns=1,
            truncate_legend=-1,
            height=800,
            style=MULTIPLE_SERIES_STYLE,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        for label, content in self.adapter_content:
            if max(content) < 0.1:
                continue
            plot.add(label, label_values(content, self.x_labels))
        return plot.render(is_unicode=True)

    def to_html(self):
        return f"""
            <h2>Adapter content</h2>
            Only adapters that are present more than 0.1% are shown. Given the 12bp
            length of the sequences used to estimate the content, values below this
            threshold are problably false positives. <br><br>
            For nanopore the the adapter mix (AMX) and ligation kit have
            overlapping adapter sequences for the bottom strand adapter.
            The ligation kit bottom strand adapter is longer however. Therefore
            the ligation kit bottom strand has two detection probes, part I and
            part II. If both are present, the bottom strand adapter is most
            likely from the ligation kit. If only part I is present, it is most
            likely from the adapter mix (AMX). <br>
            {self.plot()}
        """

    @classmethod
    def from_adapter_counter_names_and_ranges(
            cls, adapter_counter: AdapterCounter, adapter_names: Sequence[str],
            data_ranges: Sequence[Tuple[int, int]]):
        all_adapters = []
        total_sequences = adapter_counter.number_of_sequences
        for adapter, countarray in adapter_counter.get_counts():
            adapter_counts = [sum(countarray[start:stop])
                              for start, stop in data_ranges]
            total = 0
            accumulated_counts = []
            for count in adapter_counts:
                total += count
                accumulated_counts.append(total)
            all_adapters.append([count * 100 / total_sequences
                                 for count in accumulated_counts])
        return cls(stringify_ranges(data_ranges),
                   list(zip(adapter_names, all_adapters)))


@dataclasses.dataclass
class PerTileQualityReport(ReportModule):
    x_labels: Sequence[str]
    normalized_per_tile_averages: Sequence[Tuple[str, Sequence[float]]]
    tiles_2x_errors: Sequence[str]
    tiles_10x_errors: Sequence[str]
    skipped_reason: Optional[str]

    @classmethod
    def from_per_tile_quality_and_ranges(
            cls, ptq: PerTileQuality, data_ranges: Sequence[Tuple[int, int]]):
        if ptq.skipped_reason:
            return cls([], [], [], [], ptq.skipped_reason)
        average_phreds = []
        per_category_totals = [0.0 for i in range(len(data_ranges))]
        tile_counts = ptq.get_tile_counts()
        for tile, summed_errors, counts in tile_counts:
            range_averages = [
                sum(summed_errors[start:stop]) / sum(counts[start:stop])
                for start, stop in data_ranges]
            range_phreds = []
            for i, average in enumerate(range_averages):
                phred = -10 * math.log10(average)
                range_phreds.append(phred)
                # Averaging phreds takes geometric mean.
                per_category_totals[i] += phred
            average_phreds.append((tile, range_phreds))
        number_of_tiles = len(tile_counts)
        averages_per_category = [total / number_of_tiles
                                 for total in per_category_totals]
        normalized_averages = []
        tiles_2x_errors = []
        tiles_10x_errors = []
        for tile, tile_phreds in average_phreds:
            normalized_tile_phreds = [
                tile_phred - average
                for tile_phred, average in
                zip(tile_phreds, averages_per_category)
            ]
            lowest_phred = min(normalized_tile_phreds)
            if lowest_phred <= -10.0:
                tiles_10x_errors.append(str(tile))
            elif lowest_phred <= -3.0:
                tiles_2x_errors.append(str(tile))
            normalized_averages.append((str(tile), normalized_tile_phreds))
        return cls(
            x_labels=stringify_ranges(data_ranges),
            normalized_per_tile_averages=normalized_averages,
            tiles_2x_errors=tiles_2x_errors,
            tiles_10x_errors=tiles_10x_errors,
            skipped_reason=ptq.skipped_reason,
        )

    def plot(self):
        style_class = MULTIPLE_SERIES_STYLE.__class__
        red = "#FF0000"
        yellow = "#FFD700"  # actually 'Gold' which is darker and more visible.
        style = style_class(
            colors=(yellow, red) + style_class.colors
        )
        scatter_plot = pygal.Line(
            title="Deviation from geometric mean in phred units.",
            x_title="position",
            stroke=False,
            style=style,
            y_title="Normalized phred",
            truncate_legend=-1,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )

        def add_horizontal_line(name, position):
            scatter_plot.add(name,
                             [position for _ in range(len(self.x_labels))],
                             show_dots=False, stroke=True, )

        add_horizontal_line("2 times more errors", -3)
        add_horizontal_line("10 times more errors", -10)

        min_phred = -10.0
        max_phred = 0.0
        for tile, tile_phreds in self.normalized_per_tile_averages:
            min_phred = min(min_phred, *tile_phreds)
            max_phred = max(max_phred, *tile_phreds)
            scatter_plot.range = (min_phred - 1, max_phred + 1)
            if min(tile_phreds) > -3 and max(tile_phreds) < 3:
                continue
            cleaned_phreds = [{'value': phred, 'label': label}
                              if (phred > 3 or phred < -3) else None
                              for phred, label in
                              zip(tile_phreds, self.x_labels)]
            scatter_plot.add(str(tile), cleaned_phreds)

        return scatter_plot.render(is_unicode=True)

    def to_html(self) -> str:
        header = "<h2>Per tile quality</h2>"
        if self.skipped_reason:
            return header + (f"Per tile quality skipped. Reason: "
                             f"{self.skipped_reason}.")
        return header + f"""
            Tiles with more than 2 times the average error:
                {", ".join(self.tiles_2x_errors)}<br>
            Tiles with more than 10 times the average error:
                {", ".join(self.tiles_10x_errors)}<br>
            <br>
            This graph shows the deviation of each tile on each position from
            the geometric mean of all tiles at that position. The scale is
            expressed in phred units. -10 is 10 times more errors than the
            average.
            -3 is ~2 times more errors than the average. Only points that
            deviate more than 2 phred units from the average are shown. <br>
            {self.plot()}
        """


@dataclasses.dataclass
class DuplicationCounts(ReportModule):
    tracked_unique_sequences: int
    duplication_counts: Sequence[Tuple[int, int]]
    remaining_fraction: float
    estimated_duplication_fractions: Dict[str, float]

    def plot(self):
        plot = pygal.Bar(
            title="Duplication levels (%)",
            x_labels=list(self.estimated_duplication_fractions.keys()),
            x_title="Duplication counts",
            y_title="Percentage of total",
            x_label_rotation=30,
            style=ONE_SERIE_STYLE,
            **COMMON_GRAPH_OPTIONS
        )
        plot.add("",
                 [100 * fraction for fraction in
                  self.estimated_duplication_fractions.values()])
        return plot.render(is_unicode=True)

    def to_html(self):
        first_part = f"""
        All sequences are fingerprinted based on the first 16bp, the last 16bp
        and the length integer divided by 64. This means that for long
        read sequences, small indel sequencing errors will most likely not
        affect the fingerprint. <br>
        <br>
        A subsample of the fingerprints is stored to estimate the duplication
        rate. The subsample for this file consists of
        {self.tracked_unique_sequences:,} fingerprints.
        The paper describing the methodology can be found
        <a href=https://www.usenix.org/system/files/conference/atc13/atc13-xie.pdf>
        here</a>.<br>
        Estimated remaining sequences if deduplicated:
        {self.remaining_fraction:.2%}
            """
        return f"""
            <h2>Duplication percentages</h2>
            {first_part}
            <br>
            {self.plot()}
        """

    @staticmethod
    def estimated_counts_to_fractions(
            estimated_counts: Iterable[Tuple[int, int]]):
        named_slices = {
            "1": slice(1, 2),
            "2": slice(2, 3),
            "3": slice(3, 4),
            "4": slice(4, 5),
            "5": slice(5, 6),
            "6-10": slice(6, 11),
            "11-20": slice(11, 21),
            "21-30": slice(21, 31),
            "31-50": slice(31, 51),
            "51-100": slice(51, 101),
            "101-500": slice(101, 501),
            "501-1000": slice(501, 1001),
            "1001-5000": slice(1001, 5001),
            "5001-10000": slice(5001, 10_001),
            "10001-50000": slice(10_001, 50_001),
            "> 50000": slice(50_001, None),
        }
        count_array = array.array("Q", bytes(8 * 50002))
        for duplication, count in estimated_counts:
            if duplication > 50_000:
                count_array[50_001] += count * duplication
            else:
                count_array[duplication] = count * duplication
        total = sum(count_array)
        aggregated_fractions = [
            sum(count_array[slc]) / total for slc in named_slices.values()
        ]
        return dict(zip(named_slices.keys(), aggregated_fractions))

    @staticmethod
    def deduplicated_fraction(duplication_counts: Dict[int, int]):
        total_sequences = sum(duplicates * count
                              for duplicates, count in
                              duplication_counts.items())
        unique_sequences = sum(duplication_counts.values())
        return unique_sequences / total_sequences

    @classmethod
    def from_dedup_estimator(cls, dedup_est: DedupEstimator):

        tracked_unique_sequences = dedup_est.tracked_sequences
        duplication_counts = dedup_est.duplication_counts()
        duplication_categories = collections.Counter(duplication_counts)
        estimated_duplication_fractions = cls.estimated_counts_to_fractions(
            duplication_categories.items())
        deduplicated_fraction = cls.deduplicated_fraction(
            duplication_categories)
        return cls(
            tracked_unique_sequences=tracked_unique_sequences,
            duplication_counts=sorted(duplication_categories.items()),
            estimated_duplication_fractions=estimated_duplication_fractions,
            remaining_fraction=deduplicated_fraction,
        )


class OverRepresentedSequence(typing.NamedTuple):
    count: int  # type: ignore
    fraction: float
    sequence: str
    most_matches: int
    max_matches: int
    best_match: str


@dataclasses.dataclass
class OverRepresentedSequences(ReportModule):
    overrepresented_sequences: List[OverRepresentedSequence]
    max_unique_sequences: int
    collected_sequences: int
    sample_every: int
    sequence_length: int
    total_fragments: int

    def to_dict(self) -> Dict[str, Any]:
        return {"overrepresented_sequences":
                [x._asdict() for x in self.overrepresented_sequences],
                "max_unique_sequences": self.max_unique_sequences,
                "sample_every": self.sample_every,
                "collected_sequences": self.collected_sequences,
                "sequence_length": self.sequence_length,
                "total_fragments": self.total_fragments}

    def from_dict(cls, d: Dict[str, List[Dict[str, Any]]]):
        overrepresented_sequences = d["overrepresented_sequences"]
        return cls([OverRepresentedSequence(**d)
                   for d in overrepresented_sequences],
                   max_unique_sequences=d["max_unique_sequences"],
                   collected_sequences=d["collected_sequences"],
                   sample_every=d["sample_every"],
                   sequence_length=d["sequence_length"],
                   total_fragments=d["total_fragments"])  # type: ignore

    def to_html(self) -> str:
        header = "<h2>Overrepresented sequences</h2>"
        if len(self.overrepresented_sequences) == 0:
            return header + "No overrepresented sequences."
        content = io.StringIO()
        content.write(header)
        content.write(
            f"Sequences are cut into fragments of "
            f"{self.sequence_length} bp. The fragments are stored and "
            f"counted. When the fragment store is full (max "
            f"{self.max_unique_sequences:,} fragments), only sequences in the "
            f"fragment store are counted. {self.collected_sequences:,} unique "
            f"fragments were stored."
            f"1 in {self.sample_every} sequences is processed this way. "
            f"A total of {self.total_fragments:,} fragments were sampled.<br>"
        )
        content.write(
            "Fragments are stored in their canonical representation. That is "
            "either the sequence or the reverse complement, whichever has "
            "the lowest sort order. This means poly-A and poly-T sequences "
            "show up as poly-A (both are overrepresented in genomes). And "
            "illumina dark cycles (poly-G) show up as poly-C."
            "<br>")
        content.write("<table>")
        content.write("<tr><th>count</th><th>percentage</th>"
                      "<th>sequence</th><th>kmers (matched/max)</th>"
                      "<th>best match</th></tr>")
        for count, fraction, sequence, most_matches, max_matches, best_match\
                in \
                self.overrepresented_sequences:
            content.write(
                f"""<tr><td align="right">{count}</td>
                    <td align="right">{fraction * 100:.2f}</td>
                    <td>{sequence}</td>
                    <td>({most_matches}/{max_matches})</td>
                    <td>{best_match}</td></tr>""")
        content.write("</table>")
        return content.getvalue()

    @classmethod
    def from_sequence_duplication(
            cls,
            seqdup: SequenceDuplication,
            fraction_threshold: float = DEFAULT_FRACTION_THRESHOLD,
            min_threshold: int = DEFAULT_MIN_THRESHOLD,
            max_threshold: int = DEFAULT_MAX_THRESHOLD,
    ):
        overrepresented_sequences = seqdup.overrepresented_sequences(
            fraction_threshold,
            min_threshold,
            max_threshold
        )
        if overrepresented_sequences:
            def contaminant_iterator():
                for file in DEFAULT_CONTAMINANTS_FILES:
                    yield from fasta_parser(file)

            sequence_index = create_sequence_index(contaminant_iterator(),
                                                   DEFAULT_K)
        else:  # Only spend time creating sequence index when its worth it.
            sequence_index = {}
        overrepresented_with_identification = [
            OverRepresentedSequence(
                count, fraction, sequence,
                *identify_sequence(sequence, sequence_index))
            for count, fraction, sequence in overrepresented_sequences
        ]
        return cls(overrepresented_with_identification,
                   seqdup.max_unique_sequences,
                   seqdup.collected_unique_sequences,
                   seqdup.sample_every,
                   seqdup.sequence_length,
                   seqdup.total_fragments)


@dataclasses.dataclass
class NanoStatsReport(ReportModule):
    x_labels: List[str]
    time_bases: List[int]
    time_reads: List[int]
    time_active_channels: List[int]
    qual_percentages_over_time: List[List[float]]
    per_channel_bases: Dict[int, int]
    per_channel_quality: Dict[int, float]
    translocation_speed: List[int]
    skipped_reason: Optional[str] = None

    @staticmethod
    def seconds_to_hour_minute_notation(seconds: int):
        minutes = seconds // 60
        hours = minutes // 60
        minutes %= 60
        return f"{hours:02}:{minutes:02}"

    @classmethod
    def from_nanostats(cls, nanostats: NanoStats):
        if nanostats.skipped_reason:
            return cls(
                [],
                [],
                [],
                [],
                [],
                {},
                {},
                [],
                nanostats.skipped_reason
            )
        run_start_time = nanostats.minimum_time
        run_end_time = nanostats.maximum_time
        duration = run_end_time - run_start_time
        time_slots = 200
        time_per_slot = duration / time_slots
        time_interval_minutes = (math.ceil(time_per_slot) + 59) // 60
        time_interval = time_interval_minutes * 60
        time_ranges = [(start, start + time_interval)
                       for start in range(0, duration, time_interval)]
        time_slots = len(time_ranges)
        time_active_slots_sets: List[Set[int]] = [set() for _ in
                                                  range(time_slots)]
        time_bases = [0 for _ in range(time_slots)]
        time_reads = [0 for _ in range(time_slots)]
        time_qualities = [[0 for _ in range(12)] for _ in
                          range(time_slots)]
        per_channel_bases: Dict[int, int] = defaultdict(lambda: 0)
        per_channel_cumulative_error: Dict[int, float] = defaultdict(lambda: 0.0)
        translocation_speeds = [0] * 81
        for readinfo in nanostats.nano_info_iterator():
            relative_start_time = readinfo.start_time - run_start_time
            timeslot = relative_start_time // time_interval
            length = readinfo.length
            cumulative_error_rate = readinfo.cumulative_error_rate
            channel_id = readinfo.channel_id
            phred = round(
                -10 * math.log10(cumulative_error_rate / length))
            phred_index = min(phred, 47) >> 2
            time_active_slots_sets[timeslot].add(channel_id)
            time_bases[timeslot] += length
            time_reads[timeslot] += 1
            time_qualities[timeslot][phred_index] += 1
            per_channel_bases[channel_id] += length
            per_channel_cumulative_error[channel_id] += cumulative_error_rate
            read_duration = readinfo.duration
            if read_duration:
                translocation_speed = min(round(length / read_duration), 800)
                translocation_speed //= 10
                translocation_speeds[translocation_speed] += 1
        per_channel_quality: Dict[int, float] = {}
        for channel, error_rate in per_channel_cumulative_error.items():
            total_bases = per_channel_bases[channel]
            phred_score = -10 * math.log10(error_rate / total_bases)
            per_channel_quality[channel] = phred_score
        qual_percentages_over_time: List[List[float]] = [[] for _ in
                                                         range(12)]
        for quals in time_qualities:
            total = sum(quals)
            for i, q in enumerate(quals):
                qual_percentages_over_time[i].append(q / max(total, 1))
        time_active_slots = [len(s) for s in time_active_slots_sets]
        return cls(
            x_labels=[f"{cls.seconds_to_hour_minute_notation(start)}-"
                      f"{cls.seconds_to_hour_minute_notation(stop)}"
                      for start, stop in time_ranges],
            qual_percentages_over_time=qual_percentages_over_time,
            time_active_channels=time_active_slots,
            time_bases=time_bases,
            time_reads=time_reads,
            per_channel_bases=dict(sorted(per_channel_bases.items())),
            per_channel_quality=dict(sorted(per_channel_quality.items())),
            translocation_speed=translocation_speeds,
            skipped_reason=nanostats.skipped_reason
        )

    def time_bases_plot(self):
        plot = pygal.Bar(
            title="Base count over time",
            x_title="time(HH:MM)",
            y_title="base count",
            style=ONE_SERIE_STYLE,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS
        )
        plot.add("", label_values(self.time_bases, self.x_labels))
        return plot.render(is_unicode=True)

    def time_reads_plot(self):
        plot = pygal.Bar(
            title="Number of reads over time",
            x_title="time(HH:MM)",
            y_title="number of reads",
            style=ONE_SERIE_STYLE,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS
        )
        plot.add("", label_values(self.time_reads, self.x_labels))
        return plot.render(is_unicode=True)

    def time_active_channels_plot(self):
        plot = pygal.Bar(
            title="Active channels over time",
            x_title="time(HH:MM)",
            y_title="active channels",
            style=ONE_SERIE_STYLE,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS
        )
        plot.add("", label_values(self.time_active_channels, self.x_labels))
        return plot.render(is_unicode=True)

    def time_quality_distribution_plot(self):
        plot = pygal.StackedBar(
            title="Quality distribution over time",
            style=QUALITY_DISTRIBUTION_STYLE,
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="time(HH:MM)",
            y_title="fraction",
            fill=True,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        for name, serie in zip(QUALITY_SERIES_NAMES, self.qual_percentages_over_time):
            serie_filled = sum(serie) > 0.0
            plot.add(name, label_values(serie, self.x_labels),
                     show_dots=serie_filled)
        return plot.render(is_unicode=True)

    def channel_plot(self):
        plot = pygal.XY(
            title="Channel base yield and quality",
            dots_size=1,
            x_title="base yield (megabases)",
            y_title="quality (phred score)",
            stroke=False,
            style=ONE_SERIE_STYLE,
            **COMMON_GRAPH_OPTIONS
        )
        serie = []
        for channel, base_yield in self.per_channel_bases.items():
            quality = self.per_channel_quality[channel]
            serie.append(dict(value=(base_yield/1_000_000, quality),
                              label=str(channel)))
        plot.add(None, serie)
        return plot.render(is_unicode=True)

    def translocation_section(self):
        transl_speeds = self.translocation_speed
        if sum(transl_speeds) == 0:
            return """
            <h2>translocation speeds</h2>
            Duration information not available.
            """
        too_slow = transl_speeds[:35] + [0] * 55
        too_fast = [0] * 45 + transl_speeds[45:]
        normal = [0] * 35 + transl_speeds[35:45] + [0] * 35
        total = sum(transl_speeds)
        within_bounds_frac = sum(normal) / total
        too_fast_frac = sum(too_fast) / total
        too_slow_frac = sum(too_slow) / total

        plot = pygal.Bar(
            title="Translocation speed distribution",
            x_title="Translocation_speed",
            y_title="active channels",
            style=pygal.style.DefaultStyle(
                colors=(COLOR_GREEN, COLOR_RED, COLOR_RED)),
            x_labels=[str(i) for i in range(0, 800, 10)] + [">800"],
            x_labels_major_every=10,
            show_minor_x_labels=False,
            **COMMON_GRAPH_OPTIONS
        )
        plot.add("within bounds", normal)
        plot.add("too slow", too_slow)
        plot.add("too fast", too_fast)
        return f"""
        <h2>translocation speeds</h2>
        Percentage of reads within accepted bounds: {within_bounds_frac:.2%}<br>
        Percentage of reads that are too slow: {too_slow_frac:.2%}<br>
        Percentage of reads that are too fast: {too_fast_frac:.2%}<br>
        {plot.render(is_unicode=True)}
        """

    def to_html(self) -> str:
        if self.skipped_reason:
            return f"""
            <h2>Nanopore time series</h2>
            Skipped: {self.skipped_reason}
            """
        return f"""
        <h2>Nanopore time series</h2>
        <h3>Base counts over time</h3>
        {self.time_bases_plot()}
        <h3>Read counts over time</h3>
        {self.time_reads_plot()}
        <h3>Active channels over time</h3>
        {self.time_active_channels_plot()}
        <h3>Quality distribution over time</h3>
        {self.time_quality_distribution_plot()}
        <h2>Per channel base yield versus quality<h2>
        {self.channel_plot()}
        {self.translocation_section()}
        """


NAME_TO_CLASS: Dict[str, Type[ReportModule]] = {
    "summary": Summary,
    "per_position_mean_quality_and_spread": PerPositionMeanQualityAndSpread,
    "per_position_quality_distribution": PerBaseQualityScoreDistribution,
    "sequence_length_distribution": SequenceLengthDistribution,
    "per_position_base_content": PerPositionBaseContent,
    "per_position_n_content": PerPositionNContent,
    "per_sequence_gc_content": PerSequenceGCContent,
    "per_sequence_quality_scores": PerSequenceAverageQualityScores,
    "adapter_content": AdapterContent,
    "per_tile_quality": PerTileQualityReport,
    "duplication_fractions": DuplicationCounts,
    "overrepresented_sequences": OverRepresentedSequences,
    "nanopore_metrics": NanoStatsReport,
}

CLASS_TO_NAME: Dict[Type[ReportModule], str] = {
    value: key for key, value in NAME_TO_CLASS.items()}


def report_modules_to_dict(report_modules: Iterable[ReportModule]):
    return {
        CLASS_TO_NAME[type(module)]: module.to_dict()
        for module in report_modules
    }


def dict_to_report_modules(d: Dict[str, Dict[str, Any]]) -> List[ReportModule]:
    return [NAME_TO_CLASS[name].from_dict(
                NAME_TO_CLASS[name], class_dict)  # type: ignore
            for name, class_dict in d.items()]


def write_html_report(report_modules: Iterable[ReportModule],
                      html: str,
                      filename: str):
    default_config = pygal.Config()
    with open(html, "wt", encoding="utf-8") as html_file:
        html_file.write(f"""
            <html>
            <head>
                <script type="text/javascript"
                    src="https://{default_config.js[0]}"></script>
                <meta http-equiv="content-type"
                content="text/html:charset=utf-8">
                <title>{os.path.basename(filename)}: Sequali Report</title>
            </head>
            <h1>sequali report</h1>
            file: {filename}<br>
        """)
        # size: {os.stat(filename).st_size / (1024 ** 3):.2f}GiB<br>
        for module in report_modules:
            html_file.write(module.to_html())
        html_file.write("</html>")


def qc_metrics_modules(metrics: QCMetrics,
                       data_ranges: Sequence[Tuple[int, int]]
                       ) -> List[ReportModule]:
    base_count_tables = metrics.base_count_table()
    phred_count_table = metrics.phred_count_table()
    x_labels = stringify_ranges(data_ranges)
    aggregrated_base_matrix = aggregate_count_matrix(
        base_count_tables, data_ranges, NUMBER_OF_NUCS)
    aggregated_phred_matrix = aggregate_count_matrix(
        phred_count_table, data_ranges, NUMBER_OF_PHREDS)
    summary_bases = aggregate_count_matrix(
        aggregrated_base_matrix,
        [(0, len(aggregrated_base_matrix) // NUMBER_OF_NUCS)], NUMBER_OF_NUCS)
    summary_phreds = aggregate_count_matrix(
        aggregated_phred_matrix,
        [(0, len(aggregated_phred_matrix) // NUMBER_OF_PHREDS)],
        NUMBER_OF_PHREDS)
    total_bases = sum(summary_bases)
    minimum_length = 0
    total_reads = metrics.number_of_reads
    for table in table_iterator(base_count_tables, NUMBER_OF_NUCS):
        if sum(table) < total_reads:
            break
        minimum_length += 1
    a_bases = summary_bases[A]
    c_bases = summary_bases[C]
    g_bases = summary_bases[G]
    t_bases = summary_bases[T]
    gc_content = (g_bases + c_bases) / (a_bases + c_bases + g_bases + t_bases)
    return [
        Summary(
            mean_length=total_bases / total_reads,
            minimum_length=minimum_length,
            maximum_length=metrics.max_length,
            total_reads=total_reads,
            total_bases=total_bases,
            q20_bases=sum(summary_phreds[5:]),
            total_gc_fraction=gc_content),
        SequenceLengthDistribution.from_base_count_tables(
            base_count_tables, total_reads, data_ranges),
        PerBaseQualityScoreDistribution.from_phred_count_table_and_labels(
            aggregated_phred_matrix, x_labels),
        PerPositionMeanQualityAndSpread.from_phred_table_and_labels(
           aggregated_phred_matrix, x_labels),
        PerSequenceAverageQualityScores.from_qc_metrics(metrics),
        PerPositionBaseContent.from_base_count_tables_and_labels(
            aggregrated_base_matrix, x_labels),
        PerPositionNContent.from_base_count_tables_and_labels(
            aggregrated_base_matrix, x_labels),
        PerSequenceGCContent.from_qc_metrics(metrics),
    ]


def calculate_stats(
        metrics: QCMetrics,
        adapter_counter: AdapterCounter,
        per_tile_quality: PerTileQuality,
        sequence_duplication: SequenceDuplication,
        dedup_estimator: DedupEstimator,
        nanostats: NanoStats,
        adapter_names: List[str],
        graph_resolution: int = 200,
        fraction_threshold: float = DEFAULT_FRACTION_THRESHOLD,
        min_threshold: int = DEFAULT_MIN_THRESHOLD,
        max_threshold: int = DEFAULT_MAX_THRESHOLD,
) -> List[ReportModule]:
    max_length = metrics.max_length
    if max_length > 500:
        data_ranges = list(logarithmic_ranges(max_length))
    else:
        data_ranges = list(equidistant_ranges(max_length, graph_resolution))
    return [
        *qc_metrics_modules(metrics, data_ranges),
        AdapterContent.from_adapter_counter_names_and_ranges(
            adapter_counter, adapter_names, data_ranges),
        PerTileQualityReport.from_per_tile_quality_and_ranges(
            per_tile_quality, data_ranges),
        DuplicationCounts.from_dedup_estimator(dedup_estimator),
        OverRepresentedSequences.from_sequence_duplication(
            sequence_duplication,
            fraction_threshold=fraction_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold
        ),
        NanoStatsReport.from_nanostats(nanostats)
    ]
