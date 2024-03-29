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

import array

from sequali import report_modules


def test_per_position_mean_quality_one_nuc():
    # https://github.com/rhpvorderman/sequali/issues/62
    # Would cause a divide by zero error
    module = report_modules.PerPositionMeanQualityAndSpread.from_phred_table_and_labels(
        array.array('Q', [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ['1']
    )
    assert module
