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

import pytest

import cudf

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages import CompareDataframeStage
from stages import ConvMsg
from utils import TEST_DIRS
from utils import extend_df


@pytest.mark.slow
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('pipeline_batch_size', [256, 1024, 2048])
@pytest.mark.parametrize('repeat', [1, 10, 100])
def test_add_scores_stage_pipe(config, order, pipeline_batch_size, repeat):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.pipeline_batch_size = pipeline_batch_size

    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(src_file, df_type='pandas', file_type=FileTypes.Auto)
    if repeat > 1:
        input_df = extend_df(input_df, repeat)

    expected_df = input_df.rename(columns=dict(zip(input_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, order=order, columns=list(input_df.columns)))
    pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    comp_stage.get_results()["diff_rows"] == 0


@pytest.mark.slow
@pytest.mark.parametrize('repeat', [1, 2, 5])
def test_add_scores_stage_multi_segment_pipe(config, repeat):
    # Intentionally using FileSourceStage's repeat argument as this triggers a bug in #443
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(input_file, df_type='pandas', file_type=FileTypes.Auto)

    expected_df = input_df.rename(columns=dict(zip(input_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)], repeat=repeat))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, columns=list(input_df.columns)))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(AddScoresStage(config))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    comp_stage.get_results()["diff_rows"] == 0
