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

import datetime

from sequali import FastqRecordView
from sequali._qc import NanoStats


def test_nano_stats():

    view = FastqRecordView("cb1dab45-aa4c-43fc-a91e-ad0ecc92f5c9 "
                           "runid=c989c681b782549923cb0a02c95f6ec9d2534335 "
                           "read=10 ch=444 start_time=2021-09-30T11:34:08Z "
                           "flow_cell_id=PAI09842 "
                           "protocol_group_id=SS_210930_10xCDNA "
                           "sample_id=SS_A1",
                           "ACGT",
                           "AAAA")
    cumulative_error_rate = 4 * 10 ** (-(ord("A") - 33) / 10)
    # The 'Z' at the end of the start_time string is only supported from
    # python 3.11 onwards.
    tm = datetime.datetime.fromisoformat("2021-09-30T11:34:08")
    tm = tm.replace(tzinfo=datetime.timezone.utc)
    timestamp = tm.timestamp()
    nanostats = NanoStats()
    nanostats.add_read(view)
    assert nanostats.minimum_time == timestamp
    assert nanostats.maximum_time == timestamp
    nano_info_list = list(nanostats.nano_info_iterator())
    assert len(nano_info_list) == 1
    info = nano_info_list[0]
    assert info.start_time == timestamp
    assert info.channel_id == 444
    assert info.length == 4
    assert info.cumulative_error_rate == cumulative_error_rate
