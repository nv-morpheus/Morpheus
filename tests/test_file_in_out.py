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
from morpheus.io.serializers import df_to_csv
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import assert_path_exists


@pytest.mark.slow
@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
@pytest.mark.parametrize("repeat", [1, 2, 5])
def test_file_rw_pipe(tmp_path, config, output_type, flush, repeat: int):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, repeat=repeat))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, flush=flush))
    pipe.run()

    assert_path_exists(out_file)

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

    assert_path_exists(tmp_path / out_file)


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

    assert_path_exists(out_file)

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


@pytest.mark.parametrize("input_file,include_index_col",
                         [(os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv"), False),
                          (os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv"), True)])
def test_file_rw(config, tmp_path, input_file, include_index_col):
    out_file = os.path.join(tmp_path, 'results.csv')
    df = read_file_to_df(input_file, df_type='cudf')

    assert list(df.columns) == ['v1', 'v2', 'v3', 'v4']

    with open(out_file, 'w') as fh:
        fh.writelines(df_to_csv(df, include_header=True, include_index_col=include_index_col))

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_serialize_deserialize_pipe(tmp_path, config, output_type):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

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


@pytest.mark.slow
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_sfile_rw_serialize_deserialize_multi_segment_pipe(tmp_path, config, output_type):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_segment_boundary(MultiMessage)
    pipe.add_stage(SerializeStage(config))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    assert_path_exists(out_file)

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
