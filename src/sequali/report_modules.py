import array
import dataclasses
import io
import math
import sys
import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, Optional

import pygal  # type: ignore
import pygal.style  # type: ignore

from ._qc import A, C, G, N, T
from ._qc import AdapterCounter, NanoStats, PerTileQuality, QCMetrics, \
    SequenceDuplication
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE
from .sequence_identification import DEFAULT_CONTAMINANTS_FILES, DEFAULT_K, \
    create_sequence_index, identify_sequence

PHRED_TO_ERROR_RATE = [
    sum(10 ** (-p / 10) for p in range(start * 4, start * 4 + 4)) / 4
    for start in range(NUMBER_OF_PHREDS)
]

DEFAULT_FRACTION_THRESHOLD = 0.0001
DEFAULT_MIN_THRESHOLD = 100
DEFAULT_MAX_THRESHOLD = sys.maxsize

COMMON_GRAPH_OPTIONS = dict(
    truncate_label=-1,
    width=1000,
    explicit_size=True,
    disable_xml_declaration=True,
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


def logarithmic_ranges(length: int, parts: int):
    exponent = math.log(length) / math.log(parts)
    start = 0
    for i in range(1, parts + 1):
        stop = round(i ** exponent)
        length = stop - start
        if length < 1:
            continue
        yield start, stop
        start = stop


def stringify_ranges(data_ranges: Iterable[Tuple[int, int]]):
    return [
            f"{start + 1}-{stop}" if start + 1 != stop else f"{start + 1}"
            for start, stop in data_ranges
    ]


def table_iterator(count_tables: Sequence[int]) -> Iterator[memoryview]:
    table_view = memoryview(count_tables)
    for i in range(0, len(count_tables), TABLE_SIZE):
        yield table_view[i: i + TABLE_SIZE]


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


class ReportModule(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def from_dict(cls, d: Dict[str, Any]):
        return cls.__init__(**d)

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
                {self.mean_length:.2f}</td></tr>
            <tr><td>Length range (min-max)</td><td align="right">
                {self.minimum_length}-{self.maximum_length}</td></tr>
            <tr><td>total reads</td><td align="right">{self.total_reads}</td></tr>
            <tr><td>total bases</td><td align="right">{self.total_bases}</td></tr>
            <tr>
                <td>Q20 bases</td>
                <td align="right">
                    {self.q20_bases} ({self.q20_bases * 100 / self.total_bases:.2f}%)
                </td>
            </tr>
            <tr><td>GC content</td><td align="right">
                {self.total_gc_fraction * 100:.2f}%
            </td></tr>
            </table>
        """


@dataclasses.dataclass
class SequenceLengthDistribution(ReportModule):
    length_ranges: Sequence[str]
    counts: Sequence[int]

    def plot(self) -> str:
        plot = pygal.Bar(
            title="Sequence length distribution",
            x_title="sequence length",
            y_title="number of reads",
            **label_settings(self.length_ranges),
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("Length", label_values(self.counts, self.length_ranges))
        return plot.render(is_unicode=True)

    def to_html(self):
        return  f"""
            <h2>Sequence length distribution</h2>
            {self.plot()}
        """

    @classmethod
    def from_table_and_total(cls, aggregated_count_matrix: Sequence[int],
                             total_sequences: int,
                             x_labels: Sequence[str]):
        total_tabels = len(x_labels)
        base_counts = [total_sequences] + [0 for _ in range(total_tabels)]
        sequence_lengths = [0 for _ in range(total_tabels + 1)]
        for i, table in enumerate(aggregated_count_matrix):
            base_counts[i+1] = sum(table)
        previous_count = 0
        for i in range(len(x_labels), 0, -1):
            number_at_least = base_counts[i]
            sequence_lengths[i] = number_at_least - previous_count
            previous_count = number_at_least
        return cls(x_labels, sequence_lengths)


@dataclasses.dataclass
class PerBaseAverageSequenceQuality(ReportModule):
    x_labels: List[str]
    A: List[float]
    C: List[float]
    G: List[float]
    T: List[float]
    N: List[float]
    mean: List[float]

    def plot(self) -> str:
        plot = pygal.Line(
            title="Per base average sequence quality",
            dots_size=1,
            x_title="position",
            y_title="phred score",
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("A", label_values(self.A, self.x_labels))
        plot.add("C", label_values(self.C, self.x_labels))
        plot.add("G", label_values(self.G, self.x_labels))
        plot.add("T", label_values(self.T, self.x_labels))
        plot.add("mean", label_values(self.mean, self.x_labels))
        return plot.render(is_unicode=True)

    def to_html(self):
        return f"""
            <h2>Per position average quality score</h2>
            {self.plot()}
        """

    @classmethod
    def from_table_and_labels(cls, count_tables: Sequence[int], x_labels):
        total_tables = len(count_tables) // TABLE_SIZE
        mean_qualities = [0.0 for _ in range(total_tables)]
        base_qualities = [
            [0.0 for _ in range(total_tables)]
            for _ in range(NUMBER_OF_NUCS)
        ]
        for cat_index, table in enumerate(table_iterator(count_tables)):
            nuc_probs = [0.0 for _ in range(NUMBER_OF_NUCS)]
            nuc_counts = [0 for _ in range(NUMBER_OF_NUCS)]
            total_count = 0
            total_prob = 0.0
            for phred_p_value, offset in zip(
                    PHRED_TO_ERROR_RATE, range(0, TABLE_SIZE, NUMBER_OF_NUCS)
            ):
                nucs = table[offset: offset + NUMBER_OF_NUCS]
                count = sum(nucs)
                total_count += count
                total_prob += count * phred_p_value
                for i, count in enumerate(nucs):
                    nuc_counts[i] += count
                    nuc_probs[i] += phred_p_value * count
            if total_count == 0:
                continue
            mean_qualities[cat_index] = -10 * math.log(total_prob / total_count)
            for i in range(NUMBER_OF_NUCS):
                if nuc_counts[i] == 0:
                    continue
                base_qualities[i][cat_index] = -10 * math.log10(
                    nuc_probs[i] / nuc_counts[i]
                )
        return cls(
            x_labels = x_labels,
            A=base_qualities[A],
            C=base_qualities[C],
            G=base_qualities[G],
            T=base_qualities[T],
            N=base_qualities[N],
            mean=mean_qualities
        )


@dataclasses.dataclass
class PerBaseQualityScoreDistribution(ReportModule):
    x_labels: Sequence[str]
    series: Sequence[Sequence[float]]

    @classmethod
    def from_count_table_and_labels(
            cls, count_tables: Sequence[int], x_labels: Sequence[str]):
        total_tables = len(x_labels)
        quality_distribution = [
            [0.0 for _ in range(total_tables)]
            for _ in range(NUMBER_OF_PHREDS)
        ]
        for cat_index, table in enumerate(table_iterator(count_tables)):
            total_nucs = sum(table)
            for offset in range(0, TABLE_SIZE, NUMBER_OF_NUCS):
                category_nucs = sum(table[offset: offset + NUMBER_OF_NUCS])
                nuc_fraction = category_nucs / total_nucs
                quality_distribution[offset // NUMBER_OF_NUCS][
                    cat_index] = nuc_fraction
        return cls(x_labels, quality_distribution)

    def plot(self) -> str:
        dark_red = "#8B0000"  # 0-3
        red = "#ff0000"  # 4-7
        light_red = "#ff9999"  # 8-11
        white = "#FFFFFF"  # 12-15
        very_light_blue = "#e6e6ff"  # 16-19
        light_blue = "#8080ff"  # 20-23
        blue = "#0000FF"  # 24-27
        darker_blue = "#0000b3"  # 28-31
        more_darker_blue = "#000080"  # 32-35
        yet_more_darker_blue = "#00004d"  # 36-39
        almost_black_blue = "#000033"  # 40-43
        black = "#000000"  # >=44
        style = pygal.style.Style(
            colors=(
            dark_red, red, light_red, white, very_light_blue, light_blue,
            blue, darker_blue, more_darker_blue, yet_more_darker_blue,
            almost_black_blue, black)
        )
        plot = pygal.StackedLine(
            title="Per base quality distribution",
            style=style,
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="position",
            y_title="fraction",
            fill=True,
            **label_settings(self.x_labels),
            **COMMON_GRAPH_OPTIONS,
        )
        serie_names = (
            "0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27", "28-31",
            "32-35", "36-39", "40-43", ">=44")
        for name, serie in zip(serie_names, self.series):
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
    x_labels: Tuple[str] = tuple(str(x) for x in range(PHRED_MAX + 1))

    def plot(self) -> str:
        plot = pygal.Line(
            title="Per sequence quality scores",
            x_labels=range(PHRED_MAX + 1),
            width=1000,
            explicit_size=True,
            disable_xml_declaration=True,
            x_labels_major_every=3,
            show_minor_x_labels=False,
            x_title="Phred score",
            y_title="Percentage of total",
            truncate_label=-1,
        )
        total = sum(self.average_quality_counts)
        percentage_scores = [100 * score / total
                             for score in self.average_quality_counts]
        plot.add("", percentage_scores)
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per sequence average quality scores</h2>
            {self.plot()}
        """

    @classmethod
    def from_qc_metrics(cls, metrics: QCMetrics):
        return cls(metrics.phred_scores())


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
    def from_count_tables_and_labels(cls,
                                     count_tables: Sequence[int],
                                     labels: Sequence[str]):
        total_tables = len(count_tables) // TABLE_SIZE
        base_fractions = [
            [0.0 for _ in range(total_tables)]
            for _ in range(NUMBER_OF_NUCS)
        ]
        for index, table in enumerate(table_iterator(count_tables)):
            total_bases = sum(table)
            n_bases = sum(table[N::NUMBER_OF_NUCS])
            named_total = total_bases - n_bases
            if named_total == 0:
                continue
            for i in range(NUMBER_OF_NUCS):
                if i == N:
                    continue
                nuc_count = sum(table[i::NUMBER_OF_NUCS])
                base_fractions[i][index] = nuc_count / named_total
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
    def from_count_tables_and_labels(
            cls, count_tables: Sequence[int], labels: Sequence[str]):
        total_tables = len(count_tables) // TABLE_SIZE
        n_fractions = [0.0 for _ in range(total_tables)]
        for index, table in enumerate(table_iterator(count_tables)):
            total_bases = sum(table)
            if total_bases == 0:
                continue
            n_bases = sum(table[N::NUMBER_OF_NUCS])
            n_fractions[index] = n_bases / total_bases
        return cls(
            labels,
            n_fractions
        )

    def plot(self):
        plot = pygal.Line(
            title="Per position N content",
            dots_size=1,
            y_labels=[i / 10 for i in range(11)],
            x_title="position",
            y_title="fraction",
            fill=True,
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
    x_labels: Sequence[str] = tuple(str(x) for x in range(101))

    def plot(self):
        plot = pygal.Bar(
            title="Per sequence GC content",
            x_labels=self.x_labels,
            x_labels_major_every=3,
            show_minor_x_labels=False,
            x_title="GC %",
            y_title="number of reads",
            **COMMON_GRAPH_OPTIONS,
        )
        plot.add("", self.gc_content_counts)
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per sequence GC content</h2>
            {self.plot()}    
        """

    @classmethod
    def from_qc_metrics(cls, metrics: QCMetrics):
        return cls(metrics.gc_content())


@dataclasses.dataclass
class AdapterContent(ReportModule):
    x_labels: Sequence[str]
    adapter_content: Sequence[str, Sequence[float]]

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
        style_class = pygal.style.Style
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
            scatter_plot.add(name, [position for _ in range(len(self.x_labels))],
                             show_dots=False, stroke=True,)

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
                              for phred, label in zip(tile_phreds, self.x_labels)]
            scatter_plot.add(str(tile),  cleaned_phreds)

        return scatter_plot.render(is_unicode=True)

    def to_html(self) -> str:
        header = "<h2>Per tile quality</h2>"
        if self.skipped_reason:
            return header + f"Per tile quality skipper. Reason: {self.skipped_reason}."
        return header + f"""
            Tiles with more than 2 times the average error: 
                {", ".join(self.tiles_2x_errors)}<br>
            Tiles with more than 10 times the average error:
                {", ".join(self.tiles_10x_errors)}<br>
            <br>
            This graph shows the deviation of each tile on each position from
            the geometric mean of all tiles at that position. The scale is
            expressed in phred units. -10 is 10 times more errors than the average.
            -3 is ~2 times more errors than the average. Only points that
            deviate more than 2 phred units from the average are shown. <br>
            {self.plot()}
        """


class OverRepresentedSequence(typing.NamedTuple):
    count: int
    fraction: float
    sequence: str
    most_matches: int
    max_matches: int
    best_match: str


@dataclasses.dataclass
class OverRepresentedSequences(ReportModule):
    overrepresented_sequences: List[OverRepresentedSequence]
    max_unique_sequences: int

    def to_dict(self) -> Dict[str, Any]:
        return {"overrepresented_sequences":
                    [x._asdict() for x in self.overrepresented_sequences]}

    def from_dict(cls, d: Dict[str, List[Dict[str, Any]]]):
        overrepresented_sequences = d["overrepresented_sequences"]
        return cls.__init__([OverRepresentedSequence(**d)
                            for d in overrepresented_sequences])

    def to_html(self) -> str:
        header = "<h2>Overrepresented sequences</h2>"
        if len(self.overrepresented_sequences) == 0:
            return header + "No overrepresented sequences."
        content = io.StringIO()
        content.write(header)
        content.write(
            f"The first {self.max_unique_sequences} unique sequences are tracked for "
            f"duplicates. Sequences with high occurence are presented in the "
            f"table. <br>")
        content.write("Identified sequences by matched kmers. The max match is "
                "either the number of kmers in the overrepresented sequence "
                "or the number of kmers of the database sequence, whichever "
                "is fewer.")
        content.write("<table>")
        content.write("<tr><th>count</th><th>percentage</th>"
                      "<th>sequence</th><th>kmers (matched/max)</th>"
                      "<th>best match</th></tr>")
        for count, fraction, sequence, most_matches, max_matches, best_match in \
                self.overrepresented_sequences:
            content.write(
                f"""<tr><td align="right">{count}</td>
                    <td align="right">{fraction * 100:.2f}</td>
                    <td>{sequence}</td>
                    <td>({most_matches}/{max_matches})</td>
                    <td>{best_match}</td></tr>""")
        content.write("</table>")
        return content.getvalue()
