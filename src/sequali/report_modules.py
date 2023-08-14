import dataclasses
import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ReportModule(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]):
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def to_html(self) -> str:
        pass


class Summary(typing.NamedTuple):
    mean_length: float
    minimum_length: int
    maximum_length: int
    total_reads: int
    total_bases: int
    q20_bases: int
    total_gc_fraction: float


class QCMetricsReportModule(ReportModule, typing.NamedTuple):
    summary: Summary
    x_labels: List[str]
    per_position_qualities: Dict[str, List[float]]
    per_position_quality_distribution: Dict[str, List[float]]
    sequence_length_distribution: list[int]
    base_content: Dict[str, List[float]]
    per_position_n_content: Dict[str, List[float]]
    per_sequence_gc_content: List[int]
    per_sequence_quality_scores: List[int]

    def to_dict(self):
        return {
            "summary": self.summary._asdict(),
            "per_position_qualities": {
                "x_labels": self.x_labels,
                "values": self.per_position_qualities,
            },
            "per_position_quality_distribution": {
                x_labels
            }
        }

    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d["summary"])



