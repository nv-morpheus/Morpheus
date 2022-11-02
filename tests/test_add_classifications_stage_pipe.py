#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import ConvMsg
from utils import assert_file_exists_with_timeout


def test_add_classifications_stage_pipe(config, tmp_path):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    # Silly data with all false values
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_file_exists_with_timeout(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    expected = (input_data > threshold)

    # The output data will contain an additional id column that we will need to slice off
    output_data = pd.read_csv(out_file)
    idx = output_data.columns.intersection(config.class_labels)
    assert idx.to_list() == config.class_labels

    output_np = output_data[idx].to_numpy()

    assert output_np.tolist() == expected.tolist()


def test_add_classifications_stage_multi_segment_pipe(config, tmp_path):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    config.num_threads = 1

    # Silly data with all false values
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config, input_file))
    pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
    pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_file_exists_with_timeout(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    expected = (input_data > threshold)

    # The output data will contain an additional id column that we will need to slice off
    output_data = pd.read_csv(out_file)
    idx = output_data.columns.intersection(config.class_labels)
    assert idx.to_list() == config.class_labels

    output_np = output_data[idx].to_numpy()

    assert output_np.tolist() == expected.tolist()
