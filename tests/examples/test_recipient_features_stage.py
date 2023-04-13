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

import os
import typing

import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from utils import TEST_DIRS

EXPECTED_NEW_COLS = ['to_count', 'bcc_count', 'cc_count', 'total_recipients', 'data']


@pytest.mark.import_mod(
    [os.path.join(TEST_DIRS.examples_dir, 'developer_guide/2_1_real_world_phishing/recipient_features_stage.py')])
def test_recipient_features_stage_on_data(config: Config, import_mod: typing.List[typing.Any]):
    recipient_features_stage = import_mod[0]

    input_df = read_file_to_df(os.path.join(TEST_DIRS.examples_dir, 'data/email_with_addresses.jsonlines'),
                               df_type='cudf')

    config.mode = PipelineModes.NLP
    stage = recipient_features_stage.RecipientFeaturesStage(config)

    meta = MessageMeta(input_df)

    # verify that the stage is adding them, and that they are not already present

    for col in EXPECTED_NEW_COLS:
        assert col not in input_df

    results = stage.on_data(meta)

    assert results is meta
    for col in EXPECTED_NEW_COLS:
        assert col in input_df


@pytest.mark.import_mod(
    [os.path.join(TEST_DIRS.examples_dir, 'developer_guide/2_1_real_world_phishing/recipient_features_stage.py')])
def test_recipient_features_stage_needed_columns(config: Config, import_mod: typing.List[typing.Any]):
    recipient_features_stage = import_mod[0]
    config.mode = PipelineModes.NLP

    stage = recipient_features_stage.RecipientFeaturesStage(config)

    needed_cols = stage.get_needed_columns()
    assert sorted(EXPECTED_NEW_COLS) == sorted(needed_cols.keys())
