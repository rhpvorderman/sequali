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

import pytest

from sequali import FastqRecordArrayView, FastqRecordView


@pytest.mark.parametrize(["forward", "reverse", "expected"], [
    ("same", "same", True),
    ("same1", "same2", True),
    ("same with comments", "same different comments", True),
    ("same1 with comments", "same2 different comments", True),
    ("same1", "same2 with comments", True),
    ("same1 with comments", "same2", True),
    ("differnt", "diferent", False),
    ("different with comment", "diferent with comment", False),
    ("same1", "same3", False),
    ("same2", "same1", True),
    ("same2", "same5", False),
])
def test_names_are_mates(forward: str, reverse: str, expected: bool):
    forward_record = FastqRecordView(forward, "A", "A")
    reverse_record = FastqRecordView(reverse, "A", "A")
    forward_array = FastqRecordArrayView([forward_record])
    reverse_array = FastqRecordArrayView([reverse_record])
    assert forward_array.is_mate(reverse_array) is expected
