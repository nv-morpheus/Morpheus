# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""IO utilities."""

from morpheus.utils.type_aliases import DataFrameType


def filter_null_data(x: DataFrameType):
    """
    Filters out null row in a dataframe's 'data' column if it exists.

    Parameters
    ----------
    x : DataFrameType
        The dataframe to fix.
    """

    if ("data" not in x):
        return x

    return x[~x['data'].isna()]
