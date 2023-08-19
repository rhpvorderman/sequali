import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

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
