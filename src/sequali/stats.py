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
import collections
import math
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from ._qc import A, C, G, N, T
from ._qc import AdapterCounter, PerTileQuality, QCMetrics, SequenceDuplication
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE
from .sequence_identification import DEFAULT_CONTAMINANTS_FILES, DEFAULT_K, \
    create_sequence_index, identify_sequence
from .util import fasta_parser

PHRED_TO_ERROR_RATE = [
    sum(10 ** (-p / 10) for p in range(start * 4, start * 4 + 4)) / 4
    for start in range(NUMBER_OF_PHREDS)
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
        count_tables: array.array, number_of_categories: int
) -> Iterator[Tuple[int, int]]:
    max_length = len(count_tables) // TABLE_SIZE
    base_counts = array.array("Q", bytes((max_length + 1) * 8))
    base_counts[0] = max_length
    for i, table in enumerate(table_iterator(count_tables)):
        base_counts[i+1] = sum(table)
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


def stringify_ranges(data_ranges: Iterable[Tuple[int, int]]):
    return [
            f"{start + 1}-{stop}" if start + 1 != stop else f"{start + 1}"
            for start, stop in data_ranges
    ]


def cumulative_percentages(counts: Iterable[int], total: int):
    cumalitive_percentages = []
    count_sum = 0
    for count in counts:
        count_sum += count
        cumalitive_percentages.append(count_sum / total)
    return cumalitive_percentages


def normalized_per_tile_averages(
        tile_counts:  Sequence[Tuple[int, Sequence[float], Sequence[int]]],
        data_ranges: Sequence[Tuple[int, int]],
        ) -> List[Tuple[str, List[float]]]:
    if not tile_counts:
        return []
    average_phreds = []
    per_category_totals = [0.0 for i in range(len(data_ranges))]
    for tile, summed_errors, counts in tile_counts:
        range_averages = [sum(summed_errors[start:stop]) / sum(counts[start:stop])
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
    for tile, tile_phreds in average_phreds:
        normalized_tile_phreds = [
            tile_phred - average
            for tile_phred, average in zip(tile_phreds, averages_per_category)
        ]
        normalized_averages.append((str(tile), normalized_tile_phreds))
    return normalized_averages


def table_iterator(count_tables: array.ArrayType) -> Iterator[memoryview]:
    table_view = memoryview(count_tables)
    for i in range(0, len(count_tables), TABLE_SIZE):
        yield table_view[i: i + TABLE_SIZE]


def sequence_lengths(count_tables: array.ArrayType, total_reads: int):
    max_length = len(count_tables) // TABLE_SIZE
    # use bytes constructor to initialize to 0
    sequence_lengths = array.array("Q", bytes(8 * (max_length + 1)))
    base_counts = array.array("Q", bytes(8 * (max_length + 1)))
    base_counts[0] = total_reads  # all reads have at least 0 bases
    for i, table in enumerate(table_iterator(count_tables)):
        base_counts[i+1] = sum(table)
    previous_count = 0
    for i in range(max_length, 0, -1):
        number_at_least = base_counts[i]
        sequence_lengths[i] = number_at_least - previous_count
        previous_count = number_at_least
    return sequence_lengths


def aggregate_count_matrix(
        count_tables: array.ArrayType,
        data_ranges: Sequence[Tuple[int, int]]) -> array.ArrayType:
    count_view = memoryview(count_tables)
    aggregated_matrix = array.array("Q", bytes(8 * TABLE_SIZE * len(data_ranges)))
    ag_view = memoryview(aggregated_matrix)
    for cat_index, (start, stop) in enumerate(data_ranges):
        cat_offset = cat_index * TABLE_SIZE
        cat_view = ag_view[cat_offset:cat_offset + TABLE_SIZE]
        for table_index in range(start, stop):
            offset = table_index * TABLE_SIZE
            table = count_view[offset: offset + TABLE_SIZE]
            for i, count in enumerate(table):
                cat_view[i] += count
    return aggregated_matrix


def base_content(count_tables: array.ArrayType) -> List[List[float]]:
    total_tables = len(count_tables) // TABLE_SIZE
    base_fractions = [
        [0.0 for _ in range(total_tables)]
        for _ in range(NUMBER_OF_NUCS)
    ]
    for index, table in enumerate(table_iterator(count_tables)):
        total = sum(table)
        if total == 0:
            continue
        for i in range(NUMBER_OF_NUCS):
            base_fractions[i][index] = sum(table[i::NUMBER_OF_NUCS]) / total
    return base_fractions


def total_gc_fraction(count_tables: array.ArrayType) -> float:
    total_nucs = [
        sum(
            count_tables[
                i: len(count_tables): NUMBER_OF_NUCS
            ]
        )
        for i in range(NUMBER_OF_NUCS)
    ]
    at = total_nucs[A] + total_nucs[T]
    gc = total_nucs[G] + total_nucs[C]
    return gc / (at + gc)


def q20_bases(count_tables: array.ArrayType) -> int:
    q20s = 0
    for table in table_iterator(count_tables):
        q20s += sum(table[NUMBER_OF_NUCS * 5:])
    return q20s


def min_length(sequence_lengths: Sequence[int]) -> int:
    for length, count in enumerate(sequence_lengths):
        if count > 0:
            return length
    return 0


def aggregate_sequence_lengths(raw_sequence_lengths: array.ArrayType,
                               data_ranges: Iterable[Tuple[int, int]]):
    seqlength_view = memoryview(raw_sequence_lengths)[1:]
    lengths = [sum(seqlength_view[start:stop]) for start, stop in
               data_ranges]
    return [raw_sequence_lengths[0]] + lengths


def mean_qualities(count_tables: array.ArrayType) -> List[float]:
    total_tables = len(count_tables) // TABLE_SIZE
    mean_qualities = [0.0 for _ in range(total_tables)]
    for index, table in enumerate(table_iterator(count_tables)):
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
        mean_qualities[index] = -10 * math.log10(total_prob / total)
    return mean_qualities


def per_base_qualities(count_tables: array.ArrayType) -> List[List[float]]:
    total_tables = len(count_tables) // TABLE_SIZE
    base_qualities = [
        [0.0 for _ in range(total_tables)]
        for _ in range(NUMBER_OF_NUCS)
    ]
    for cat_index, table in enumerate(table_iterator(count_tables)):
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


def adapter_counts(adapter_counter: AdapterCounter,
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
    return list(zip(adapter_counter.adapters, all_adapters))


def estimate_duplication_counts(
        duplication_counts: Dict[int, int],
        total_sequences: int,
        gathered_sequences: int) -> Dict[int, int]:
    estimated_counts: Dict[int, int] = {}
    for duplicates, number_of_occurences in duplication_counts.items():
        chance_of_random_draw = duplicates / total_sequences
        chance_of_random_not_draw = 1 - chance_of_random_draw
        chance_of_not_draw_at_gathering = chance_of_random_not_draw ** gathered_sequences  # noqa: E501
        chance_of_draw_at_gathering = 1 - chance_of_not_draw_at_gathering
        estimated_counts[duplicates] = round(number_of_occurences / chance_of_draw_at_gathering)  # noqa: E501
    return estimated_counts


def duplication_fractions(
        duplication_counts: Dict[int, int]) -> Dict[int, float]:
    total_sequences = sum(duplicates * count
                          for duplicates, count in duplication_counts.items())
    return {duplicates: count / total_sequences for duplicates, count
            in duplication_counts.items()}


def deduplicated_fraction(duplication_counts: Dict[int, int]):
    total_sequences = sum(duplicates * count
                          for duplicates, count in duplication_counts.items())
    unique_sequences = sum(duplication_counts.values())
    return unique_sequences / total_sequences


def aggregate_duplication_counts(sequence_duplication: SequenceDuplication):
    named_slices = {
        "1": slice(1, 2),
        "2": slice(2, 3),
        "3": slice(3, 4),
        "4": slice(4, 5),
        "5": slice(5, 6),
        "6-10": slice(6, 11),
        "11-50": slice(11, 51),
        "51-100": slice(51, 101),
        "101-500": slice(101, 501),
        "501-1000": slice(501, 1001),
        "1001-5000": slice(1001, 5001),
        "5001-10000": slice(5001, 10_001),
        "10001-50000": slice(10_001, 50_001),
        "> 50000": slice(50_001, None),
    }
    duplication_counts = sequence_duplication.duplication_counts(50_001)
    aggregated_counts = [
        sum(duplication_counts[slc]) for slc in named_slices.values()
    ]
    return list(named_slices.keys()), aggregated_counts


def calculate_stats(
        metrics: QCMetrics,
        adapter_counter: AdapterCounter,
        per_tile_quality: PerTileQuality,
        sequence_duplication: SequenceDuplication,
        graph_resolution: int = 200) -> Dict[str, Any]:
    count_table = metrics.count_table()

    data_ranges = (
        list(equidistant_ranges(metrics.max_length, graph_resolution))
        if metrics.max_length < 500 else
        list(base_weighted_categories(count_table, graph_resolution))
    )
    aggregated_table = aggregate_count_matrix(count_table, data_ranges)
    total_bases = sum(aggregated_table)
    total_reads = metrics.number_of_reads
    seq_lengths = sequence_lengths(count_table, total_reads)
    x_labels = stringify_ranges(data_ranges)
    pbq = per_base_qualities(aggregated_table)
    bc = base_content(aggregated_table)
    per_tile_phreds = normalized_per_tile_averages(
        per_tile_quality.get_tile_counts(), data_ranges)
    rendered_tiles = []
    warn_tiles = []
    error_tiles = []
    good_tiles = []
    for tile, tile_phreds in per_tile_phreds:
        tile_minimum = min(tile_phreds)
        tile_maximum = max(tile_phreds)
        if tile_minimum > -2 and tile_maximum < 2:
            good_tiles.append(tile)
            continue
        rendered_tiles.append((tile, tile_phreds))
        if tile_minimum < -10 or tile_maximum > 10:
            error_tiles.append(tile)
        else:
            warn_tiles.append(tile)
    overrepresented_sequences = sequence_duplication.overrepresented_sequences()

    if overrepresented_sequences:
        def contaminant_iterator():
            for file in DEFAULT_CONTAMINANTS_FILES:
                yield from fasta_parser(file)

        sequence_index = create_sequence_index(contaminant_iterator(), DEFAULT_K)
    else:  # Only spend time creating sequence index when its worth it.
        sequence_index = {}
    overrepresented_with_identification = [
        (count, fraction, sequence, *identify_sequence(sequence, sequence_index))
        for count, fraction, sequence in overrepresented_sequences
    ]
    sequence_counts = sequence_duplication.sequence_counts()
    duplication_counts = collections.Counter(sequence_counts.values())
    estimated_duplication_counts = estimate_duplication_counts(
        duplication_counts, sequence_duplication.number_of_sequences,
        sequence_duplication.stopped_collecting_at)
    duplicated_labels, duplicated_counts = \
        aggregate_duplication_counts(sequence_duplication)
    return {
        "summary": {
            "mean_length": total_bases / total_reads,
            "minimum_length": min_length(seq_lengths),
            "maximum_length": metrics.max_length,
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
                "N": pbq[N],
            },
        },
        "sequence_length_distribution": {
            "x_labels": ["0"] + x_labels,
            "values": aggregate_sequence_lengths(seq_lengths, data_ranges)
        },
        "base_content": {
            "x_labels": x_labels,
            "values": {
                "A": bc[A],
                "C": bc[C],
                "G": bc[G],
                "T": bc[T],
                "N": bc[N],
            },
        },
        "per_sequence_gc_content": {
            "x_labels": [str(i) for i in range(101)],
            "values": list(metrics.gc_content()),
        },
        "per_sequence_quality_scores": {
            "x_labels": [str(i) for i in range(PHRED_MAX + 1)],
            "values": list(metrics.phred_scores()),
        },
        "adapter_content": {
            "x_labels": x_labels,
            "values": adapter_counts(adapter_counter, data_ranges)

        },
        "per_tile_quality": {
            "skipped_reason": per_tile_quality.skipped_reason,
            "good_tiles": good_tiles,
            "warn_tiles": warn_tiles,
            "error_tiles": error_tiles,
            "normalized_per_tile_averages_for_problematic_tiles": rendered_tiles,
            "x_labels": x_labels,
        },
        "overrepresented_sequences": overrepresented_with_identification,
        "duplication_counts": {
            "remaining_percentage":
                deduplicated_fraction(estimated_duplication_counts) * 100,
            "values": duplicated_counts,
            "x_labels": duplicated_labels,
        }
    }
