from typing import Any, Dict, List, Sequence, Tuple

import pygal  # type: ignore

from ._qc import A, C, G, N, T
from ._qc import PHRED_MAX


def per_tile_graph(per_tile_phreds: List[Tuple[str, List[float]]],
                   x_labels: List[str]) -> str:

    scatter_plot = pygal.Line(
        title="Sequence length distribution",
        x_labels=x_labels,
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
        stroke=False,
    )

    for tile, tile_phreds in per_tile_phreds:
        scatter_plot.add(str(tile),  tile_phreds)

    return scatter_plot.render(is_unicode=True)


def per_base_quality_plot(per_base_qualities: Dict[str, Sequence[float]],
                          x_labels: Sequence[str]) -> str:
    plot = pygal.Line(
        title="Per base sequence quality",
        dots_size=1,
        x_labels=x_labels,
        truncate_label=-1,
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    plot.add("A", per_base_qualities["A"])
    plot.add("C", per_base_qualities["C"])
    plot.add("G", per_base_qualities["G"])
    plot.add("T", per_base_qualities["T"])
    plot.add("mean", per_base_qualities["mean"])
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


def base_content_plot(base_content: Dict[str, Sequence[float]],
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
    plot.add("G", base_content["G"], fill=True)
    plot.add("C", base_content["C"], fill=True)
    plot.add("A", base_content["A"], fill=True)
    plot.add("T", base_content["T"], fill=True)
    plot.add("N", base_content["N"], fill=True)
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


def adapter_content_plot(adapter_content: Sequence[Tuple[str, Sequence[float]]],
                         x_labels: Sequence[str],) -> str:
    plot = pygal.Line(
        title="Adapter content (%)",
        x_labels=x_labels,
        range=(0.0, 100.0),
        width=1000,
        explicit_size=True,
        disable_xml_declaration=True,
    )
    for label, content in adapter_content:
        plot.add(label, content)
    return plot.render(is_unicode=True)


def html_report(data: Dict[str, Any]):
    summary = data["summary"]
    ptq = data["per_tile_quality"]
    skipped_reason = ptq["skipped_reason"]
    if skipped_reason:
        ptq_content = f"Per tile quality skipped. Reason: {skipped_reason}"
    else:
        ptq_content = per_tile_graph(
            data["per_tile_quality"]["normalized_per_tile_averages"],
            data["per_tile_quality"]["x_labels"]
        )
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
    </html>
    """
