import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Optional

import pygal  # type: ignore
import pygal.style  # type: ignore

from ._qc import PHRED_MAX

COMMON_GRAPH_OPTIONS = dict(
    truncate_label=-1,
    width=1000,
    explicit_size=True,
    disable_xml_declaration=True,
)


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
    length_ranges: List[str]
    counts: List[int]

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


@dataclasses.dataclass
class PerBaseQualityScoreDistribution(ReportModule):
    x_labels: List[str]
    series: List[List[str]]

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
    x_labels: Tuple[str] = tuple(range(PHRED_MAX + 1))

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


@dataclasses.dataclass
class PerPositionNContent(ReportModule):
    x_labels: Sequence[str]
    n_content: Sequence[float]

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

    def plot(self):
        plot = pygal.Bar(
            title="Per sequence GC content",
            x_labels=range(101),
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


@dataclasses.dataclass
class PerSequenceQualityScores(ReportModule):
    quality_counts: Sequence[int]

    def plot(self):
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
        total = sum(self.quality_counts)
        percentage_scores = [100 * count / total
                             for count in self.quality_counts]
        plot.add("", percentage_scores)
        return plot.render(is_unicode=True)

    def to_html(self) -> str:
        return f"""
            <h2>Per sequence quality scores</h2>
            {self.plot()}
        """


@dataclasses.dataclass
class AdapterContent(ReportModule):
    x_labels: Sequence[str]
    adapter_content: Dict[str, Sequence[float]]

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


@dataclasses.dataclass
class PerTileQuality(ReportModule):
    x_labels: Sequence[str]
    normalized_per_tile_averages: Dict[int, Sequence[float]]
    tiles_2x_errors: Sequence[str]
    tiles_10x_errors: Sequence[str]
    skipped_reason: Optional[str]

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
