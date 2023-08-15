import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict


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
