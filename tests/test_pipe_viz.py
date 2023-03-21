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

import imghdr
import os

import pytest

from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages.conv_msg import ConvMsg
from utils import TEST_DIRS
from utils import assert_path_exists


@pytest.fixture(scope="function")
def viz_pipeline(config, tmp_path):
    """
    Creates a quick pipeline.
    """

    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    # Silly data with all false values
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_stage(AddClassificationsStage(config))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    return pipe


def test_call_before_run(viz_pipeline: Pipeline, tmp_path):

    # Test is necessary to ensure run() is called first. See issue #230
    viz_file = os.path.join(tmp_path, 'pipeline.png')

    with pytest.raises(RuntimeError):

        viz_pipeline.visualize(viz_file, rankdir="LR")


def test_png(viz_pipeline: Pipeline, tmp_path):

    viz_file = os.path.join(tmp_path, 'pipeline.png')

    # Call pipeline run first
    viz_pipeline.run()

    viz_pipeline.visualize(viz_file, rankdir="LR")

    # Verify that the output file exists and is a valid png file
    assert_path_exists(viz_file)
    assert imghdr.what(viz_file) == 'png'
