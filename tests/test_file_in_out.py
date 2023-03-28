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

import filecmp
import os

import numpy as np
import pytest

from morpheus.common import FileTypes
from morpheus.config import CppConfig
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import write_df_to_file
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from utils import TEST_DIRS
from utils import assert_df_equal
from utils import assert_path_exists


@pytest.mark.slow
@pytest.mark.parametrize("input_type", ["csv", "jsonlines", "parquet"])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
@pytest.mark.parametrize("flush", [False, True], ids=["no_flush", "flush"])
@pytest.mark.parametrize("repeat", [1, 2, 5], ids=["repeat1", "repeat2", "repeat5"])
def test_file_rw_pipe(tmp_path, config, input_type, output_type, flush, repeat: int):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.{}'.format(input_type))
    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, flush=flush))
    pipe.run()

    assert_path_exists(out_file)

    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)

    # Repeat the input data
    validation_data = np.tile(validation_data, (repeat, 1))

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    elif output_type == "json" or output_type == "jsonlines":  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values
    elif output_type == "parquet":
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values
    else:
        assert False, "Unknown file extension"

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


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

    assert_path_exists(tmp_path / out_file)


@pytest.mark.slow
@pytest.mark.parametrize("input_type", ["csv", "jsonlines", "parquet"])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_multi_segment_pipe(tmp_path, config, input_type, output_type):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.{}'.format(input_type))
    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    if (input_type == "parquet"):
        CppConfig.set_should_use_cpp(False)

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)

    if output_type == "csv":
        # The output data will contain an additional id column that we will need to slice off
        output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
        output_data = output_data[:, 1:]
    else:  # assume json
        df = read_file_to_df(out_file, file_type=FileTypes.Auto)
        output_data = df.values

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("input_file",
                         [
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv")
                         ])
def test_file_rw_index_pipe(tmp_path, config, input_file):
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, include_index_col=False))
    pipe.run()

    assert_path_exists(out_file)

    validation_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    validation_data = np.loadtxt(validation_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)

    # Somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == validation_data.tolist()


@pytest.mark.parametrize("input_file,extra_kwargs",
                         [(os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"), {
                             "include_header": True, "include_index_col": False
                         }), (os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv"), {
                             "include_header": True
                         }), (os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines"), {})],
                         ids=["CSV", "CSV_ID", "JSON"])
def test_file_roundtrip(use_cpp, tmp_path, input_file, extra_kwargs):

    # Output file should be same type as input
    out_file = os.path.join(tmp_path, 'results{}'.format(os.path.splitext(input_file)[1]))

    # Read the dataframe
    df = read_file_to_df(input_file, df_type='cudf')

    # Write the dataframe
    write_df_to_file(df=df, file_name=out_file, **extra_kwargs)

    assert_path_exists(out_file)

    assert filecmp.cmp(input_file, out_file)


@pytest.mark.parametrize("input_file",
                         [
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.jsonlines"),
                             os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.parquet")
                         ],
                         ids=["CSV", "CSV_ID", "JSON", "PARQUET"])
def test_read_cpp_compare(input_file):

    # First read with python
    CppConfig.set_should_use_cpp(False)
    df_python = read_file_to_df(input_file, df_type='cudf')

    # Now with C++
    CppConfig.set_should_use_cpp(True)
    df_cpp = read_file_to_df(input_file, df_type='cudf')

    assert assert_df_equal(df_python, df_cpp)
