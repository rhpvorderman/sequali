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

import os
import typing
from typing import Iterator, Literal, Optional

DEFAULT_ADAPTER_FILE = os.path.join(os.path.dirname(__file__),
                                    "adapters", "adapter_list.tsv")


class Adapter(typing.NamedTuple):
    name: str
    sequencing_technology: str
    sequence: str
    sequence_position: Literal["start", "end"]


def adapters_from_file(adapter_file: str,
                       sequencing_technology: Optional[str] = None
                       ) -> Iterator[Adapter]:
    with open(adapter_file, "rt") as seqfile:
        for line in seqfile:
            line = line.strip()
            if not line:
                continue  # ignore empty lines
            if line.startswith("#"):
                continue  # Use # as a comment character
            name, seqtech, sequence, position = line.split("\t")
            if position not in ("start", "end"):
                raise ValueError(f"position must be start or end, got '{position}'.")
            if (sequencing_technology is None or
                    (seqtech == sequencing_technology or
                     seqtech == "all")):
                yield Adapter(name, seqtech, sequence, position)  # type: ignore
