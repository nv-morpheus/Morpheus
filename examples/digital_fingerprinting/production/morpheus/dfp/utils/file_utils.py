# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from datetime import datetime
from datetime import timezone

import fsspec

iso_date_regex = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"T(?P<hour>\d{1,2})(:|_)(?P<minute>\d{1,2})(:|_)(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})?Z")


def date_extractor(file_object: fsspec.core.OpenFile, filename_regex: re.Pattern):

    assert isinstance(file_object, fsspec.core.OpenFile)

    file_path = file_object.path

    # Match regex with the pathname since that can be more accurate
    match = filename_regex.search(file_path)

    if (match):
        # Convert the regex match
        groups = match.groupdict()

        if ("microsecond" in groups):
            groups["microsecond"] = int(float(groups["microsecond"]) * 1000000)

        groups = {key: int(value) for key, value in groups.items()}

        groups["tzinfo"] = timezone.utc

        ts_object = datetime(**groups)
    else:
        # Otherwise, fallback to the file modified (created?) time
        ts_object = file_object.fs.modified(file_object.path)

    return ts_object
