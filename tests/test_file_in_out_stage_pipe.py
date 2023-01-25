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

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from utils import TEST_DIRS
from utils import assert_file_exists_with_timeout


@pytest.mark.slow
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
@pytest.mark.parametrize("repeat", [1, 2, 5])
def test_file_rw_pipe(tmp_path, config, output_type, repeat: int):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_file_exists_with_timeout(out_file)

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    # Repeat the input data
    input_data = np.tile(input_data, (repeat, 1))

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


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.usefixtures("chdir_tmpdir")
def test_to_file_no_path(tmp_path, config):
    """
    Test to ensure issue #48 is fixed
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = "test.csv"

    assert os.path.realpath(os.curdir) == tmp_path.as_posix()

    assert not os.path.exists(tmp_path / out_file)
    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_file_exists_with_timeout(tmp_path / out_file)


@pytest.mark.slow
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_multi_segment_pipe(tmp_path, config, output_type):
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
