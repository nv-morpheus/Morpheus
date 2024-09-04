#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.cpu_mode
@pytest.mark.parametrize("use_deserialize", [False, True])
@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_pipe(tmp_path: str,
                      config: Config,
                      dataset: DatasetManager,
                      output_type: str,
                      flush: bool,
                      use_deserialize: bool):
    """
    Test the flush functionality of the WriteToFileStage.
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, f'results.{output_type}')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))

    if use_deserialize:
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(SerializeStage(config))

    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, flush=flush))
    pipe.run()

    assert os.path.exists(out_file)

    expected_df = dataset['filter_probs.csv']
    actual_df = dataset.get_df(out_file, no_cache=True)
    dataset.assert_compare_df(expected_df, actual_df)
