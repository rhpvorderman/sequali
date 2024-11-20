# Copyright (C) 2023 Leiden University Medical Center
# This file is part of Sequali
#
# Sequali is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Sequali is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Sequali.  If not, see <https://www.gnu.org/licenses/

from ._qc import A, C, G, N, T
from ._qc import (
    AdapterCounter, BamParser, FastqParser, FastqRecordArrayView,
    FastqRecordView, OverrepresentedSequences, PerTileQuality, QCMetrics,
)
from ._qc import NUMBER_OF_NUCS, NUMBER_OF_PHREDS, PHRED_MAX, TABLE_SIZE
from ._version import __version__


__all__ = [
    "A", "C", "G", "N", "T",
    "AdapterCounter",
    "BamParser",
    "FastqParser",
    "FastqRecordView",
    "FastqRecordArrayView",
    "PerTileQuality",
    "QCMetrics",
    "OverrepresentedSequences",
    "NUMBER_OF_NUCS",
    "NUMBER_OF_PHREDS",
    "PHRED_MAX",
    "TABLE_SIZE",
    "__version__"
]
