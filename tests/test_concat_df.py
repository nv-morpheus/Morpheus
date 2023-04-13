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

from dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.utils import concat_df


def test_concat_df(config: Config, dataset: DatasetManager):
    meta = MessageMeta(dataset["filter_probs.csv"])
    messages = [
        meta,
        MultiMessage(meta=meta, mess_offset=0, mess_count=10),
        MultiMessage(meta=meta, mess_offset=10, mess_count=10)
    ]

    results = concat_df.concat_dataframes(messages)

    pdf = dataset.pandas["filter_probs.csv"]

    expected_df = pd.concat([pdf, pdf[0:10], pdf[10:]])
    dataset.assert_df_equal(results, expected_df)
