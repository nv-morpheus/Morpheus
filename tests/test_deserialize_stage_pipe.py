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

import cudf

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataframeStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import TEST_DIRS
from utils import assert_df_equal
from utils import assert_results
from utils import duplicate_df_index
from utils import duplicate_df_index_rand


def test_fixing_non_unique_indexes(use_cpp):

    df = read_file_to_df(os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv'),
                         file_type=FileTypes.Auto,
                         df_type="cudf")

    # Set 2 ids equal to others
    df = duplicate_df_index_rand(df, count=2)

    meta = MessageMeta(df.copy())

    assert not meta.has_sliceable_index(), "Need to start with a non-sliceable index"

    # When processing the dataframe, a warning should be generated when there are non-unique IDs
    with pytest.warns(RuntimeWarning):

        DeserializeStage.process_dataframe(meta, 5, ensure_sliceable_index=False)

        assert not meta.has_sliceable_index()
        assert "_index_" not in meta.df.columns

    assert assert_df_equal(meta.df, df)

    DeserializeStage.process_dataframe(meta, 5, ensure_sliceable_index=True)

    assert meta.has_sliceable_index()
    assert "_index_" in meta.df.columns


@pytest.mark.slow
@pytest.mark.parametrize("dup_index", [False, True])
def test_deserialize_pipe(config, dup_index: bool):
    """
    End to end test for DeserializeStage
    """
    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(src_file, df_type='pandas')
    expected_df = input_df.copy(deep=True)

    if dup_index:
        input_df = duplicate_df_index(input_df, {8: 7})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config, include=[r'^v\d+$']))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.slow
@pytest.mark.parametrize("dup_index", [False, True])
def test_deserialize_multi_segment_pipe(config, dup_index: bool):
    """
    End to end test for FileSourceStage & WriteToFileStage across mulitiple segments
    """
    src_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    input_df = read_file_to_df(src_file, df_type='pandas')
    expected_df = input_df.copy(deep=True)

    if dup_index:
        input_df = duplicate_df_index(input_df, {8: 7})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config, include=[r'^v\d+$']))
    comp_stage = pipe.add_stage(CompareDataframeStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())
