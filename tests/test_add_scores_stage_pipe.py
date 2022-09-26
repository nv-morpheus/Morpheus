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
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import ConvMsg


def test_add_scores_stage_pipe(config, tmp_path):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config))
    pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert os.path.exists(out_file)

    expected = np.loadtxt(input_file, delimiter=",", skiprows=1)

    # The output data will contain an additional id column that we will need to slice off
    # also somehow 0.7 ends up being 0.7000000000000001
    output_data = pd.read_csv(out_file)
    idx = output_data.columns.intersection(config.class_labels)
    assert idx.to_list() == config.class_labels

    output_np = np.around(output_data[idx].to_numpy(), 2)

    assert output_np.tolist() == expected.tolist()


def test_add_scores_stage_multi_segment_pipe(config, tmp_path):
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']

    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config))
    pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(AddScoresStage(config))
    pipe.add_segment_boundary(MultiResponseProbsMessage)
    pipe.add_stage(SerializeStage(config, include=["^{}$".format(c) for c in config.class_labels]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert os.path.exists(out_file)

    expected = np.loadtxt(input_file, delimiter=",", skiprows=1)

    # The output data will contain an additional id column that we will need to slice off
    # also somehow 0.7 ends up being 0.7000000000000001
    output_data = pd.read_csv(out_file)
    idx = output_data.columns.intersection(config.class_labels)
    assert idx.to_list() == config.class_labels

    output_np = np.around(output_data[idx].to_numpy(), 2)

    assert output_np.tolist() == expected.tolist()
