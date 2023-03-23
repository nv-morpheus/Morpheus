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
import pandas as pd
import pytest

import cudf

from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataframeStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import TEST_DIRS
from utils import assert_results
from utils import extend_df


def build_expected(df: pd.DataFrame, threshold: float):
    """
    Takes a copy of `df` and apply the threshold
    """
    expected_df = df.copy(deep=True)
    return expected_df[expected_df.max(axis=1) >= threshold]


def _test_filter_detections_stage_pipe(config, copy=True, order='K', pipeline_batch_size=256, repeat=1):
    config.pipeline_batch_size = pipeline_batch_size

    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(src_file, df_type='pandas')

    if repeat > 1:
        input_df = extend_df(input_df, repeat)

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, order=order, columns=list(input_df.columns)))
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, build_expected(input_df, threshold)))
    pipe.run()

    assert_results(comp_stage.get_results())


def _test_filter_detections_stage_multi_segment_pipe(config, copy=True):
    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(src_file, df_type='pandas')

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataframeStage(config, build_expected(input_df, threshold)))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.slow
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('pipeline_batch_size', [256, 1024, 2048])
@pytest.mark.parametrize('repeat', [1, 10, 100])
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_stage_pipe(config, order, pipeline_batch_size, repeat, do_copy):
    return _test_filter_detections_stage_pipe(config, do_copy, order, pipeline_batch_size, repeat)


@pytest.mark.slow
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_stage_multi_segment_pipe(config, do_copy):
    return _test_filter_detections_stage_multi_segment_pipe(config, do_copy)
