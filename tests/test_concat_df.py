# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pandas as pd

import cudf

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.utils import concat_df
from utils import assert_df_equal


def test_concat_df(config, filter_probs_df):
    meta = MessageMeta(filter_probs_df.copy(deep=True))
    messages = [
        meta,
        MultiMessage(meta=meta, mess_offset=0, mess_count=10),
        MultiMessage(meta=meta, mess_offset=10, mess_count=10)
    ]

    results = concat_df.concat_dataframes(messages)

    pdf = filter_probs_df.copy(deep=True)
    if isinstance(pdf, cudf.DataFrame):
        pdf = pdf.to_pandas()

    expected_df = pd.concat([pdf, pdf[0:10], pdf[10:]])
    assert_df_equal(results, expected_df)
