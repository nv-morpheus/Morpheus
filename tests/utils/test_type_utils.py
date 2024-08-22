#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing
from unittest.mock import MagicMock

import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.utils.type_utils import get_df_class
from morpheus.utils.type_utils import get_df_pkg
from morpheus.utils.type_utils import is_cudf_type


def test_get_df_class(config: Config):
    if config.execution_mode == ExecutionMode.GPU:
        expected = cudf.DataFrame
    else:
        expected = pd.DataFrame

    assert get_df_class(config) is expected


def test_get_df_pkg(config: Config):
    if config.execution_mode == ExecutionMode.GPU:
        expected = cudf
    else:
        expected = pd

    assert get_df_pkg(config) is expected


@pytest.mark.parametrize(
    "obj, expected",
    [
        (cudf.DataFrame(), True),
        (cudf.Series(), True),
        (cudf.Index([]), True),
        (cudf.RangeIndex(0), True),
        (pd.DataFrame(), False),
        (pd.Series(), False),
        (pd.Index([]), False),
        (pd.RangeIndex(0), False),
        (None, False),
        (0, False),
        ("test", False),
    ],
    ids=[
        "cudf.DataFrame",
        "cudf.Series",
        "cudf.Index",
        "cudf.RangeIndex",
        "pd.DataFrame",
        "pd.Series",
        "pd.Index",
        "pd.RangeIndex",
        "None",
        "int",
        "str"
    ],
)
def test_is_cudf_type(obj: typing.Any, expected: bool):
    assert is_cudf_type(obj) == expected
