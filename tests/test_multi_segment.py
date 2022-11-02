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

from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from utils import TEST_DIRS
from utils import assert_file_exists_with_timeout


# Adapted from fil_in_out_stage -- used for testing multi-segment error conditions
def test_linear_boundary_stages(tmp_path, config, output_type='json'):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_file_exists_with_timeout(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()


def test_multi_segment_bad_data_type(tmp_path, config, output_type='json'):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    try:
        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=input_file))
        pipe.add_segment_boundary(int)
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
        pipe.run()
        assert (False)
    except Exception as e:
        return e

    assert_file_exists_with_timeout(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()
