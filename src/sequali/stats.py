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
import sys
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from . import __version__
from ._qc import A, C, G, N, T
from ._qc import AdapterCounter, NanoStats, PerTileQuality, QCMetrics, \
    SequenceDuplication
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE
from .sequence_identification import DEFAULT_CONTAMINANTS_FILES, DEFAULT_K, \
    create_sequence_index, identify_sequence
from .util import fasta_parser

PHRED_TO_ERROR_RATE = [
    sum(10 ** (-p / 10) for p in range(start * 4, start * 4 + 4)) / 4
    for start in range(NUMBER_OF_PHREDS)
]

DEFAULT_FRACTION_THRESHOLD = 0.0001
DEFAULT_MIN_THRESHOLD = 100
DEFAULT_MAX_THRESHOLD = sys.maxsize


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


def base_content(count_tables: array.ArrayType) -> Dict[str, List[float]]:
    total_tables = len(count_tables) // TABLE_SIZE
    base_counts = [
        [0 for _ in range(total_tables)]
        for _ in range(NUMBER_OF_NUCS)
    ]
    for index, table in enumerate(table_iterator(count_tables)):
        for i in range(NUMBER_OF_NUCS):
            base_counts[i][index] = sum(table[i::NUMBER_OF_NUCS])
    named_totals = []
    totals = []
    for i in range(total_tables):
        named_total = (base_counts[A][i] + base_counts[C][i] +
                       base_counts[G][i] + base_counts[T][i])
        total = named_total + base_counts[N][i]
        named_totals.append(named_total)
        totals.append(total)
    return {
        "A": [count / total for count, total in
              zip(base_counts[A], named_totals)],
        "C": [count / total for count, total in
              zip(base_counts[C], named_totals)],
        "G": [count / total for count, total in
              zip(base_counts[G], named_totals)],
        "T": [count / total for count, total in
              zip(base_counts[T], named_totals)],
        "N": [count / total for count, total in
              zip(base_counts[N], totals)],
    }


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


def per_base_quality_distribution(count_tables: array.ArrayType) -> List[List[float]]:
    total_tables = len(count_tables) // TABLE_SIZE
    quality_distribution = [
        [0.0 for _ in range(total_tables)]
        for _ in range(NUMBER_OF_PHREDS)
    ]
    for cat_index, table in enumerate(table_iterator(count_tables)):
        total_nucs = sum(table)
        for offset in range(0, TABLE_SIZE, NUMBER_OF_NUCS):
            category_nucs = sum(table[offset: offset + NUMBER_OF_NUCS])
            nuc_fraction = category_nucs / total_nucs
            quality_distribution[offset // NUMBER_OF_NUCS][cat_index] = nuc_fraction
    return quality_distribution


def adapter_counts(adapter_counter: AdapterCounter,
                   adapter_names: List[str],
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
    return list(zip(adapter_names, all_adapters))


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


def deduplicated_fraction(duplication_counts: Dict[int, int]):
    total_sequences = sum(duplicates * count
                          for duplicates, count in duplication_counts.items())
    unique_sequences = sum(duplication_counts.values())
    return unique_sequences / total_sequences


def estimated_counts_to_fractions(estimated_counts: Dict[int, int]):
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
    for duplication, count in estimated_counts.items():
        if duplication > 50_000:
            count_array[50_001] += count * duplication
        else:
            count_array[duplication] = count * duplication
    total = sum(count_array)
    aggregated_fractions = [
        sum(count_array[slc]) / total for slc in named_slices.values()
    ]
    return list(named_slices.keys()), aggregated_fractions


def nanostats_time_series(nanostats: NanoStats, divisor = 600):
    run_start_time = nanostats.minimum_time
    run_end_time = nanostats.maximum_time
    duration = run_end_time - run_start_time
    time_slots = (duration + divisor - 1) // divisor
    time_active_slots_sets = [set() for _ in range(time_slots)]
    time_bases = [0 for _ in range(time_slots)]
    time_reads = [0 for _ in range(time_slots)]
    time_qualities = [[0 for _ in range(12)] for _ in range(time_slots)]
    for readinfo in nanostats.nano_info_list():
        relative_start_time = readinfo.start_time - run_start_time
        timeslot = relative_start_time // divisor
        length = readinfo.length
        phred = round(-10 * math.log10(readinfo.cumulative_error_rate / length))
        phred_index = min(phred, 47) >> 2
        time_active_slots_sets[timeslot].add(readinfo.channel_id)
        time_bases[timeslot] += length
        time_reads[timeslot] += 1
        time_qualities[timeslot][phred_index] += 1
    qual_percentages = [[] for _ in range(12)]
    for quals in time_qualities:
        total = sum(quals)
        for i, q in enumerate(quals):
            qual_percentages[i].append(q / max(total, 1))
    time_active_slots = [len(s) for s in time_active_slots_sets]
    return qual_percentages, time_active_slots, time_bases, time_reads


def calculate_stats(
        metrics: QCMetrics,
        adapter_counter: AdapterCounter,
        per_tile_quality: PerTileQuality,
        sequence_duplication: SequenceDuplication,
        nanostats: NanoStats,
        adapter_names: List[str],
        graph_resolution: int = 200,
        fraction_threshold: float = DEFAULT_FRACTION_THRESHOLD,
        min_threshold: int = DEFAULT_MIN_THRESHOLD,
        max_threshold: int = DEFAULT_MAX_THRESHOLD,
) -> Dict[str, Any]:
    count_table = metrics.count_table()

    data_ranges = (
        list(equidistant_ranges(metrics.max_length, graph_resolution))
        if metrics.max_length < 500 else
        list(logarithmic_ranges(metrics.max_length, graph_resolution))
    )
    aggregated_table = aggregate_count_matrix(count_table, data_ranges)
    total_bases = sum(aggregated_table)
    total_reads = metrics.number_of_reads
    seq_lengths = sequence_lengths(count_table, total_reads)
    x_labels = stringify_ranges(data_ranges)
    pbq = per_base_qualities(aggregated_table)
    bc = base_content(aggregated_table)
    n_content = bc.pop("N")
    per_tile_phreds = normalized_per_tile_averages(
        per_tile_quality.get_tile_counts(), data_ranges)
    rendered_tiles = []
    tiles_2x_errors = []
    tiles_10x_errors = []
    tiles_average_errors = []
    for tile, tile_phreds in per_tile_phreds:
        tile_minimum = min(tile_phreds)
        tile_maximum = max(tile_phreds)
        if tile_minimum > -3 and tile_maximum < 3:
            tiles_average_errors.append(tile)
            continue
        rendered_tiles.append((tile, tile_phreds))
        if tile_minimum < -10 or tile_maximum > 10:
            tiles_10x_errors.append(tile)
        else:
            tiles_2x_errors.append(tile)
    overrepresented_sequences = sequence_duplication.overrepresented_sequences(
        threshold_fraction=fraction_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold
    )

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
    duplication_counts = collections.Counter(
        sequence_duplication.duplication_counts())
    estimated_duplication_counts = estimate_duplication_counts(
        duplication_counts, sequence_duplication.number_of_sequences,
        sequence_duplication.stopped_collecting_at)
    duplicated_labels, duplicated_fractions = \
        estimated_counts_to_fractions(estimated_duplication_counts)
    return {
        "meta": {
            "sequali_version": __version__,
            "max_unique_sequences": sequence_duplication.max_unique_sequences,
        },
        "summary": {
            "mean_length": total_bases / total_reads,
            "minimum_length": min_length(seq_lengths),
            "maximum_length": metrics.max_length,
            "total_reads": total_reads,
            "total_bases": total_bases,
            "q20_bases": q20_bases(aggregated_table),
            "total_gc_fraction": total_gc_fraction(aggregated_table),
        },
        "per_position_qualities": {
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
        "per_position_quality_distribution": {
            "x_labels": x_labels,
            "values": dict(zip([
                "0-3",
                "4-7",
                "8-11",
                "12-15",
                "16-19",
                "20-23",
                "24-27",
                "28-31",
                "32-35",
                "36-39",
                "40-43",
                ">=44"
            ], per_base_quality_distribution(aggregated_table))),
        },
        "sequence_length_distribution": {
            "x_labels": ["0"] + x_labels,
            "values": aggregate_sequence_lengths(seq_lengths, data_ranges)
        },
        "base_content": {
            "x_labels": x_labels,
            "values": bc,
        },
        "per_position_n_content": {
            "x_labels": x_labels,
            "values": n_content,
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
            "values": adapter_counts(adapter_counter, adapter_names, data_ranges)

        },
        "per_tile_quality": {
            "skipped_reason": per_tile_quality.skipped_reason,
            "tiles_average_errors": tiles_average_errors,
            "tiles_2x_errors": tiles_2x_errors,
            "tiles_10x_errors": tiles_10x_errors,
            "normalized_per_tile_averages_for_problematic_tiles": rendered_tiles,
            "x_labels": x_labels,
        },
        "overrepresented_sequences": overrepresented_with_identification,
        "duplication_fractions": {
            "remaining_fraction":
                deduplicated_fraction(estimated_duplication_counts),
            "values": duplicated_fractions,
            "x_labels": duplicated_labels,
        }
    }
