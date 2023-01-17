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

from morpheus._lib.file_types import FileTypes
from morpheus._lib.filter_source import FilterSource
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import ConvMsg


@pytest.mark.slow
@pytest.mark.parametrize('use_conv_msg', [True, False])
@pytest.mark.parametrize('do_copy', [True, False])
@pytest.mark.parametrize('threshold', [0.1, 0.5, 0.8])
@pytest.mark.parametrize('field_name', ['v1', 'v2', 'v3', 'v4'])
def test_filter_column(config, tmp_path, use_conv_msg, do_copy, threshold, field_name):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.csv')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
    pipe.add_stage(DeserializeStage(config))

    # When `use_conv_msg` is true, ConvMsg will convert messages to MultiResponseProbs,
    # when false, the filter stage will receive instances of MultiMessage
    if use_conv_msg:
        pipe.add_stage(ConvMsg(config, empty_probs=True))

    pipe.add_stage(
        FilterDetectionsStage(config,
                              threshold=threshold,
                              copy=do_copy,
                              data_source=FilterSource.DATAFRAME,
                              field_name=field_name))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))
    pipe.run()

    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)

    # The output data will contain an additional id column that we will need to slice off
    # also somehow 0.7 ends up being 0.7000000000000001
    output_data = np.around(output_data[:, 1:], 2)

    expected_df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='pandas')
    expected_df = expected_df[expected_df[field_name] > threshold]

    assert output_data.tolist() == expected_df.to_numpy().tolist()
