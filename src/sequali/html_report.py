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

from ._qc import PHRED_MAX

COMMON_GRAPH_OPTIONS = dict(
    truncate_label=-1,
    width=1000,
    explicit_size=True,
    disable_xml_declaration=True,
)


def label_values(values: Sequence[Any], labels: Sequence[Any]):
    if len(values) != len(labels):
        raise ValueError("labels and values should have the same length")
    return [{"value": value, "label": label} for value, label
            in zip(values, labels)]


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


def per_tile_graph(per_tile_phreds: List[Tuple[str, List[float]]],
                   x_labels: List[str]) -> str:
    # Set different colors for the error and warn lines
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
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )

    def add_horizontal_line(name, position):
        scatter_plot.add(name, [position for _ in range(len(x_labels))],
                         show_dots=False, stroke=True,)

    add_horizontal_line("2 times more errors", -3)
    add_horizontal_line("10 times more errors", -10)

    min_phred = -10.0
    max_phred = 0.0
    for tile, tile_phreds in per_tile_phreds:
        min_phred = min(min_phred, *tile_phreds)
        max_phred = max(max_phred, *tile_phreds)
        scatter_plot.range = (min_phred - 1, max_phred + 1)
        if min(tile_phreds) > -3 and max(tile_phreds) < 3:
            continue
        cleaned_phreds = [{'value': phred, 'label': label}
                          if (phred > 3 or phred < -3) else None
                          for phred, label in zip(tile_phreds, x_labels)]
        scatter_plot.add(str(tile),  cleaned_phreds)

    return scatter_plot.render(is_unicode=True)


def per_position_quality_plot(per_position_qualities: Dict[str, Sequence[float]],
                              x_labels: Sequence[str]) -> str:
    plot = pygal.Line(
        title="Per base average sequence quality",
        dots_size=1,
        x_title="position",
        y_title="phred score",
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("A", label_values(per_position_qualities["A"], x_labels))
    plot.add("C", label_values(per_position_qualities["C"], x_labels))
    plot.add("G", label_values(per_position_qualities["G"], x_labels))
    plot.add("T", label_values(per_position_qualities["T"], x_labels))
    plot.add("mean", label_values(per_position_qualities["mean"], x_labels))
    return plot.render(is_unicode=True)


def per_position_quality_distribution_plot(
        per_position_quality_distribution: Dict[str, Sequence[float]],
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
    plot = pygal.StackedLine(
        title="Per base quality distribution",
        style=style,
        dots_size=1,
        y_labels=[i / 10 for i in range(11)],
        x_title="position",
        y_title="fraction",
        fill=True,
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )
    for name, serie in per_position_quality_distribution.items():
        serie_filled = sum(serie) > 0.0
        plot.add(name, label_values(serie, x_labels), show_dots=serie_filled)
    return plot.render(is_unicode=True)


def sequence_length_distribution_plot(sequence_lengths: Sequence[int],
                                      x_labels: Sequence[str]) -> str:
    plot = pygal.Bar(
        title="Sequence length distribution",
        x_title="sequence length",
        y_title="number of reads",
        **label_settings(x_labels),
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
    plot = pygal.StackedLine(
        title="Base content",
        style=style,
        dots_size=1,
        y_labels=[i / 10 for i in range(11)],
        x_title="position",
        y_title="fraction",
        fill=True,
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("G", label_values(base_content["G"], x_labels))
    plot.add("C", label_values(base_content["C"], x_labels))
    plot.add("A", label_values(base_content["A"], x_labels))
    plot.add("T", label_values(base_content["T"], x_labels))
    return plot.render(is_unicode=True)


def n_content_plot(n_content: Sequence[float],
                   x_labels: Sequence[str]) -> str:
    plot = pygal.Line(
        title="N content",
        dots_size=1,
        y_labels=[i / 10 for i in range(11)],
        x_title="position",
        y_title="fraction",
        fill=True,
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )
    plot.add("N", label_values(n_content, x_labels))
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
    plot = pygal.Line(
        title="Adapter content (%)",
        range=(0.0, 100.0),
        x_title="position",
        y_title="%",
        legend_at_bottom=True,
        legend_at_bottom_columns=1,
        truncate_legend=-1,
        height=800,
        **label_settings(x_labels),
        **COMMON_GRAPH_OPTIONS,
    )
    for label, content in adapter_content:
        if max(content) < 0.1:
            continue
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
        overrepresented_sequences: Sequence[Tuple[int, float, str, int, int, str]],
        max_unique_sequences: int,
) -> str:
    if not overrepresented_sequences:
        return "No overrepresented sequences."
    table = io.StringIO()
    table.write(
        f"The first {max_unique_sequences} unique sequences are tracked for "
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
    max_unique_sequences = data["meta"]["max_unique_sequences"]
    if skipped_reason:
        ptq_content = f"Per tile quality skipped. Reason: {skipped_reason}"
    else:
        ptq_text = f"""
        Tiles with more than 2 times the average error:
        {", ".join(ptq["tiles_2x_errors"])}<br>
        Tiles with more than 10 times the average error:
        {", ".join(ptq["tiles_10x_errors"])}<br>
        <br>
        This graph shows the deviation of each tile on each position from
        the geometric mean of all tiles at that position. The scale is
        expressed in phred units. -10 is 10 times more errors than the average.
        -3 is ~2 times more errors than the average. Only points that
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
    <h2>Sequence length distribution</h2>
    {sequence_length_distribution_plot(
        data["sequence_length_distribution"]["values"],
        data["sequence_length_distribution"]["x_labels"],
    )}
    <h2>Per position quality score distribution</h2>
    {per_position_quality_distribution_plot(
        data["per_position_quality_distribution"]["values"],
        data["per_position_quality_distribution"]["x_labels"])}
    <h2>Per position average quality score</h2>
    {per_position_quality_plot(data["per_position_qualities"]["values"],
                               data["per_position_qualities"]["x_labels"], )}
    <h2>Per sequence quality scores</h2>
    {per_sequence_quality_scores_plot(data["per_sequence_quality_scores"]["values"])}
    <h2>Per Tile Quality</h2>
    {ptq_content}
    <h2>Per position base content</h2>
    {base_content_plot(data["base_content"]["values"],
                       data["base_content"]["x_labels"])}
    <h2>Per position N content</h2>
    {n_content_plot(data["per_position_n_content"]["values"],
                    data["per_position_n_content"]["x_labels"])}
    <h2>Per sequence GC content</h2>
    {per_sequence_gc_content_plot(data["per_sequence_gc_content"]["values"])}
    <h2>Adapter content plot</h2>
    Only adapters that are present more than 0.1% are shown. Given the 12bp
    length of the sequences used to estimate the content, values below this
    threshold are problably false positives. <br>
    {adapter_content_plot(data["adapter_content"]["values"],
                          data["adapter_content"]["x_labels"])}
    <h2>Duplication percentages</h2>
    This estimates the fraction of the duplication based on the first
    {max_unique_sequences} unique sequences. <br>
    Estimated remaining sequences if deduplicated:
        {data["duplication_fractions"]["remaining_fraction"]:.2%}
    <br>
    {duplication_percentages_plot(data["duplication_fractions"]["values"],
                                  data["duplication_fractions"]["x_labels"])}
    <h2>Overrepresented sequences</h2>
    {overrepresented_sequences_content(data["overrepresented_sequences"],
                                       max_unique_sequences)}
    </html>
    """
