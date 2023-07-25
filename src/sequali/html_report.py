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
import io
from typing import Any, Dict, List, Sequence, Tuple

import pygal  # type: ignore
import pygal.style  # type: ignore

from ._qc import MAX_UNIQUE_SEQUENCES, PHRED_MAX

COMMON_GRAPH_OPTIONS = dict(
    truncate_label=-1,
    width=1000,
    explicit_size=True,
    disable_xml_declaration=True,
)


def label_values(values: Sequence[Any], labels: Sequence[Any]):
    return [{"value": value, "label": label} for value, label
            in zip(values, labels)]


def per_tile_graph(per_tile_phreds: List[Tuple[str, List[float]]],
                   x_labels: List[str]) -> str:
    # Set different colors for the error and warn lines
    style_class = pygal.style.Style
    red = "#FF0000"
    yellow = "#FFD700"  # actually 'Gold' which is darker and more visible.
    style = style_class(
        colors=(yellow, yellow, red, red) + style_class.colors
    )
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    scatter_plot = pygal.Line(
        title="Deviation from geometric mean in phred units.",
        x_labels=simple_x_labels,
        x_title="position",
        stroke=False,
        style=style,
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        y_title="Normalized phred",
        **COMMON_GRAPH_OPTIONS,
    )

    def add_horizontal_line(name, position):
        scatter_plot.add(name, [position for _ in range(len(x_labels))],
                         show_dots=False, stroke=True,)

    add_horizontal_line("warn", -2)
    add_horizontal_line("warn", 2)
    add_horizontal_line("error", -10)
    add_horizontal_line("error", 10)

    min_phred = -10.0
    max_phred = 10.0
    for tile, tile_phreds in per_tile_phreds:
        cleaned_phreds = [{'value': phred, 'label': label}
                          if (phred > 2 or phred < -2) else None
                          for phred, label in zip(tile_phreds, x_labels)]
        scatter_plot.add(str(tile),  cleaned_phreds)
        min_phred = min(min_phred, *tile_phreds)
        max_phred = max(max_phred, *tile_phreds)
    scatter_plot.range = (min_phred - 1, max_phred + 1)
    return scatter_plot.render(is_unicode=True)


def per_base_quality_plot(per_base_qualities: Dict[str, Sequence[float]],
                          x_labels: Sequence[str]) -> str:
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    plot = pygal.Line(
        title="Per base average sequence quality",
        dots_size=1,
        x_labels=simple_x_labels,
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        x_title="position",
        y_title="phred score",
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("A", label_values(per_base_qualities["A"], x_labels))
    plot.add("C", label_values(per_base_qualities["C"], x_labels))
    plot.add("G", label_values(per_base_qualities["G"], x_labels))
    plot.add("T", label_values(per_base_qualities["T"], x_labels))
    plot.add("mean", label_values(per_base_qualities["mean"], x_labels))
    return plot.render(is_unicode=True)


def per_base_quality_distribution_plot(
        per_base_quality_distribution: Dict[str, Sequence[float]],
        x_labels: Sequence[str]) -> str:
    dark_red = "#8B0000"                # 0-3
    red = "#ff0000"                     # 4-7
    light_red = "#ff9999"               # 8-11
    white = "#FFFFFF"                   # 12-15
    very_light_blue = "#e6e6ff"         # 16-19
    light_blue = "#8080ff"              # 20-23
    blue = "#0000FF"                    # 24-27
    darker_blue = "#0000b3"             # 28-31
    more_darker_blue = "#000080"        # 32-35
    yet_more_darker_blue = "#00004d"    # 36-39
    almost_black_blue = "#000033"       # 40-43
    black = "#000000"                   # >=44
    style = pygal.style.Style(
        colors=(dark_red, red, light_red, white, very_light_blue, light_blue,
                blue, darker_blue, more_darker_blue, yet_more_darker_blue,
                almost_black_blue, black)
    )
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    plot = pygal.StackedLine(
        title="Per base quality distribution",
        style=style,
        dots_size=1,
        x_labels=simple_x_labels,
        y_labels=[i / 10 for i in range(11)],
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        x_title="position",
        y_title="fraction",
        fill=True,
        **COMMON_GRAPH_OPTIONS,
    )
    for name, serie in per_base_quality_distribution.items():
        serie_filled = sum(serie) > 0.0
        plot.add(name, label_values(serie, x_labels), show_dots=serie_filled)
    return plot.render(is_unicode=True)


def sequence_length_distribution_plot(sequence_lengths: Sequence[int],
                                      x_labels: Sequence[str]) -> str:
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    plot = pygal.Bar(
        title="Sequence length distribution",
        x_labels=simple_x_labels,
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        x_title="sequence length",
        y_title="number of reads",
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("Length", label_values(sequence_lengths, x_labels))
    return plot.render(is_unicode=True)


def base_content_plot(base_content: Dict[str, Sequence[float]],
                      x_labels: Sequence[str]) -> str:
    style_class = pygal.style.Style
    red = "#DC143C"  # Crimson
    dark_red = "#8B0000"  # DarkRed
    blue = "#00BFFF"  # DeepSkyBlue
    dark_blue = "#1E90FF"  # DodgerBlue
    black = "#000000"
    style = style_class(
        colors=(red, dark_red, blue, dark_blue, black)
    )
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    plot = pygal.StackedLine(
        title="Base content",
        style=style,
        dots_size=1,
        x_labels=simple_x_labels,
        y_labels=[i / 10 for i in range(11)],
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        x_title="position",
        y_title="fraction",
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("G", label_values(base_content["G"], x_labels), fill=True)
    plot.add("C", label_values(base_content["C"], x_labels), fill=True)
    plot.add("A", label_values(base_content["A"], x_labels), fill=True)
    plot.add("T", label_values(base_content["T"], x_labels), fill=True)
    plot.add("N", label_values(base_content["N"], x_labels), fill=True)
    return plot.render(is_unicode=True)


def per_sequence_gc_content_plot(gc_content: Sequence[int]) -> str:
    plot = pygal.Bar(
        title="Per sequence GC content",
        x_labels=range(101),
        x_labels_major_every=3,
        show_minor_x_labels=False,
        x_title="GC %",
        y_title="number of reads",
        **COMMON_GRAPH_OPTIONS,
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
        x_labels_major_every=3,
        show_minor_x_labels=False,
        x_title="Phred score",
        y_title="Percentage of total",
        truncate_label=-1,
    )
    total = sum(per_sequence_quality_scores)
    percentage_scores = [100 * score / total
                         for score in per_sequence_quality_scores]
    plot.add("", percentage_scores)
    return plot.render(is_unicode=True)


def adapter_content_plot(adapter_content: Sequence[Tuple[str, Sequence[float]]],
                         x_labels: Sequence[str],) -> str:
    simple_x_labels = [label.split("-")[0] for label in x_labels]
    plot = pygal.Line(
        title="Adapter content (%)",
        x_labels=simple_x_labels,
        range=(0.0, 100.0),
        x_labels_major_every=round(len(x_labels) / 30),
        show_minor_x_labels=False,
        x_title="position",
        y_title="%",
        legend_at_bottom=True,
        legend_at_bottom_columns=1,
        truncate_legend=-1,
        height=800,
        **COMMON_GRAPH_OPTIONS,
    )
    for label, content in adapter_content:
        plot.add(label, label_values(content, x_labels))
    return plot.render(is_unicode=True)


def duplication_percentages_plot(duplication_fractions: Sequence[float],
                                 x_labels: Sequence[str]) -> str:
    plot = pygal.Bar(
        title="Duplication levels (%)",
        x_labels=x_labels,
        x_title="Duplication counts",
        y_title="Percentage of total",
        x_label_rotation=30,
        **COMMON_GRAPH_OPTIONS
    )
    plot.add("", [100 * fraction for fraction in duplication_fractions])
    return plot.render(is_unicode=True)


def overrepresented_sequences_content(
        overrepresented_sequences: Sequence[Tuple[int, float, str, int, int, str]]
) -> str:
    if not overrepresented_sequences:
        return "No overrepresented sequences."
    table = io.StringIO()
    table.write(
        f"The first {MAX_UNIQUE_SEQUENCES} unique sequences are tracked for "
        f"duplicates. Sequences with high occurence are presented in the "
        f"table. <br>")
    table.write("Identified sequences by matched kmers. The max match is "
                "either the number of kmers in the overrepresented sequence "
                "or the number of kmers of the database sequence, whichever "
                "is fewer.")
    table.write("<table>")
    table.write("<tr><th>count</th><th>percentage</th>"
                "<th>sequence</th><th>kmers (matched/max)</th>"
                "<th>best match</th></tr>")
    for count, fraction, sequence, most_matches, max_matches, best_match in \
            overrepresented_sequences:
        table.write(
            f"""<tr><td align="right">{count}</td>
                <td align="right">{fraction * 100:.2f}</td>
                <td>{sequence}</td>
                <td>({most_matches}/{max_matches})</td>
                <td>{best_match}</td></tr>""")
    table.write("</table>")
    return table.getvalue()


def html_report(data: Dict[str, Any]):
    summary = data["summary"]
    ptq = data["per_tile_quality"]
    skipped_reason = ptq["skipped_reason"]
    if skipped_reason:
        ptq_content = f"Per tile quality skipped. Reason: {skipped_reason}"
    else:
        ptq_text = """
        This graph shows the deviation of each tile on each position from
        the geometric mean of all tiles at that position. The scale is
        expressed in phred units. -10 is 10 times more errors than the average.
        -2 is 1.58 times more errors than the average. Only points that
        deviate more than 2 phred units from the average are shown. <br>
        """
        ptq_graph = per_tile_graph(
            data["per_tile_quality"][
                "normalized_per_tile_averages_for_problematic_tiles"],
            data["per_tile_quality"]["x_labels"]
        )
        ptq_content = ptq_text + ptq_graph
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
                           data["per_base_qualities"]["x_labels"], )}
    </html>
    <h2>Quality score distribution</h2>
    {per_base_quality_distribution_plot(
        data["per_base_quality_distribution"]["values"],
        data["per_base_quality_distribution"]["x_labels"])}
    <h2>Sequence length distribution</h2>
    {sequence_length_distribution_plot(
        data["sequence_length_distribution"]["values"],
        data["sequence_length_distribution"]["x_labels"],
    )}
    <h2>Base content</h2>
    {base_content_plot(data["base_content"]["values"],
                       data["base_content"]["x_labels"])}
    <h2>Per sequence GC content</h2>
    {per_sequence_gc_content_plot(data["per_sequence_gc_content"]["values"])}
    <h2>Per sequence quality scores</h2>
    {per_sequence_quality_scores_plot(data["per_sequence_quality_scores"]["values"])}
    <h2>Adapter content plot</h2>
    {adapter_content_plot(data["adapter_content"]["values"],
                          data["adapter_content"]["x_labels"])}
    <h2>Per Tile Quality</h2>
    {ptq_content}
    <h2>Duplication percentages</h2>
    This estimates the fraction of the duplication based on the first
    {MAX_UNIQUE_SEQUENCES} unique sequences. <br>
    Estimated remaining sequences if deduplicated:
        {data["duplication_fractions"]["remaining_fraction"]:.2%}
    <br>
    {duplication_percentages_plot(data["duplication_fractions"]["values"],
                                  data["duplication_fractions"]["x_labels"])}
    <h2>Overrepresented sequences</h2>
    {overrepresented_sequences_content(data["overrepresented_sequences"])}
    </html>
    """
