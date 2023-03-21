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

from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from stages import ConvMsg
from utils import TEST_DIRS
from utils import assert_path_exists
from utils import extend_data
from utils import get_column_names_from_file


def _test_filter_detections_stage_pipe(config, tmp_path, copy=True, order='K', pipeline_batch_size=256, repeat=1):
    config.pipeline_batch_size = pipeline_batch_size

    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    input_cols = get_column_names_from_file(src_file)
    if repeat > 1:
        input_file = os.path.join(tmp_path, 'input.csv')
        extend_data(src_file, input_file, repeat)
    else:
        input_file = src_file

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config, order=order, columns=input_cols))
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)

    # The output data will contain an additional id column that we will need to slice off
    # also somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data[:, 1:], 2)

    expected = input_data[np.any(input_data >= threshold, axis=1), :]
    assert output_data.tolist() == expected.tolist()


def _test_filter_detections_stage_multi_segment_pipe(config, tmp_path, copy=True):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    threshold = 0.75

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(ConvMsg(config))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(FilterDetectionsStage(config, threshold=threshold, copy=copy))
    pipe.add_segment_boundary(MultiResponseMessage)
    pipe.add_stage(SerializeStage(config))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)

    # The output data will contain an additional id column that we will need to slice off
    # also somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data[:, 1:], 2)

    expected = input_data[np.any(input_data >= threshold, axis=1), :]
    assert output_data.tolist() == expected.tolist()


@pytest.mark.slow
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('pipeline_batch_size', [256, 1024, 2048])
@pytest.mark.parametrize('repeat', [1, 10, 100])
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_stage_pipe(config, tmp_path, order, pipeline_batch_size, repeat, do_copy):
    return _test_filter_detections_stage_pipe(config, tmp_path, do_copy, order, pipeline_batch_size, repeat)


@pytest.mark.slow
@pytest.mark.parametrize('do_copy', [True, False])
def test_filter_detections_stage_multi_segment_pipe(config, tmp_path, do_copy):
    return _test_filter_detections_stage_multi_segment_pipe(config, tmp_path, do_copy)
