#!/usr/bin/env python
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

import os

import numpy as np
import pytest

from morpheus._lib.common import FilterSource
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataframeStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import TEST_DIRS
from utils import assert_results_equal


@pytest.mark.slow
@pytest.mark.parametrize('use_conv_msg', [True, False])
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, use_conv_msg, do_copy, threshold, field_name):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(input_file, df_type='cudf')

    expected_df = input_df.to_pandas()
    expected_df = expected_df[expected_df[field_name] > threshold]

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [input_df]))
    pipe.add_stage(DeserializeStage(config))

    # When `use_conv_msg` is true, ConvMsg will convert messages to MultiResponseProbs,
    # when false, the filter stage will receive instances of MultiMessage
    if use_conv_msg:
        pipe.add_stage(ConvMsg(config, empty_probs=True))

    pipe.add_stage(
        FilterDetectionsStage(config,
                              threshold=threshold,
                              copy=do_copy,
                              filter_source=FilterSource.DATAFRAME,
                              field_name=field_name))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    assert_results_equal(comp_stage.get_results())
