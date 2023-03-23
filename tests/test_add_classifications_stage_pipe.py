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
import typing

import pandas as pd
import pytest

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import TEST_DIRS
from utils import assert_results


def build_expected(df: pd.DataFrame, threshold: float, class_labels: typing.List[str]):
    """
    Generate the expected output of an add class by filtering by a threshold and applying the class labels
    """
    df = (df > threshold)
    # Replace input columns with the class labels
    return df.rename(columns=dict(zip(df.columns, class_labels)))


@pytest.mark.slow
@pytest.mark.use_pandas
def test_add_classifications_stage_pipe(config, filter_probs_df):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    threshold = 0.75

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    expected_df = (filter_probs_df > threshold)
    # Replace input columns with the class labels
    expected_df = expected_df.rename(columns=dict(zip(expected_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, build_expected(filter_probs_df, threshold, config.class_labels)))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.slow
@pytest.mark.use_pandas
def test_add_classifications_stage_multi_segment_pipe(config, filter_probs_df):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    threshold = 0.75

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    comp_stage = pipe.add_stage(
        CompareDataFrameStage(config, build_expected(filter_probs_df, threshold, config.class_labels)))
    pipe.run()

    assert_results(comp_stage.get_results())
