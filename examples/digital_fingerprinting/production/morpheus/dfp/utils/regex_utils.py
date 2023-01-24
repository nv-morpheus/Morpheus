# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

iso_date_regex_pattern = (
    r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"T(?P<hour>\d{1,2})(:|_)(?P<minute>\d{1,2})(:|_)(?P<second>\d{1,2})(?P<microsecond>\.\d{1,6})?Z")

iso_date_regex = re.compile(iso_date_regex_pattern)
