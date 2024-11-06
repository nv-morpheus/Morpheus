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

import types

import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
import pandas as pd


@pytest.mark.parametrize(
    "output_file",
    [
        "/tmp/output.json",  # "/tmp/output.csv",
  # "/tmp/output.parquet"
    ])
@pytest.mark.gpu_and_cpu_mode
def test_write_to_file_stage_pipe(config, df_pkg: types.ModuleType, dataset: DatasetManager, output_file: str) -> None:
    """
    Test WriteToFileStage with different output formats (JSON, CSV, Parquet)
    """

    filter_probs_df = dataset['filter_probs.csv']
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))
    pipe.run()

    # Load the output file and compare with the input dataframe
    if output_file.endswith(".json"):
        with open(output_file, 'r') as f:
            output_df = pd.concat([pd.read_json(line) for line in f], ignore_index=True)
    elif output_file.endswith(".csv"):
        output_df = df_pkg.read_csv(output_file)
    elif output_file.endswith(".parquet"):
        output_df = df_pkg.read_parquet(output_file)
    else:
        raise ValueError(f"Unsupported file format: {output_file}")

    dataset.assert_compare_df(filter_probs_df, output_df)
