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

import typing

import pandas as pd
import pytest

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import assert_results


def build_expected(df: pd.DataFrame, threshold: float, class_labels: typing.List[str]):
    """
    Generate the expected output of an add class by filtering by a threshold and applying the class labels
    """
    df = (df > threshold)
    # Replace input columns with the class labels
    return df.rename(columns=dict(zip(df.columns, class_labels)))


@pytest.mark.use_cudf
def test_add_classifications_stage_pipe(config, filter_probs_df):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1
    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, filter_probs_df))
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, build_expected(filter_probs_df.to_pandas(), threshold, config.class_labels)))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.use_cudf
def test_add_classifications_stage_multi_segment_pipe(config, filter_probs_df):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1
    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, filter_probs_df))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, build_expected(filter_probs_df.to_pandas(), threshold, config.class_labels)))
    pipe.run()

    assert_results(comp_stage.get_results())
