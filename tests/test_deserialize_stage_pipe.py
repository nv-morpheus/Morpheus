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

import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from utils import assert_results
from utils.dataset_manager import DatasetManager


@pytest.mark.use_cudf
def test_fixing_non_unique_indexes(use_cpp: bool, dataset: DatasetManager):
    # Set 2 ids equal to others
    df = dataset.dup_index(dataset["filter_probs.csv"], count=2)

    meta = MessageMeta(df.copy())

    assert not meta.has_sliceable_index(), "Need to start with a non-sliceable index"

    # When processing the dataframe, a warning should be generated when there are non-unique IDs
    with pytest.warns(RuntimeWarning):

        DeserializeStage.process_dataframe(meta, 5, ensure_sliceable_index=False)

        assert not meta.has_sliceable_index()
        assert "_index_" not in meta.df.columns

    assert dataset.assert_df_equal(meta.df, df)

    DeserializeStage.process_dataframe(meta, 5, ensure_sliceable_index=True)

    assert meta.has_sliceable_index()
    assert "_index_" in meta.df.columns


@pytest.mark.use_cudf
@pytest.mark.parametrize("dup_index", [False, True])
def test_deserialize_pipe(config: Config, dataset: DatasetManager, dup_index: bool):
    """
    End to end test for DeserializeStage
    """

    filter_probs_df = dataset["filter_probs.csv"]
    if dup_index:
        filter_probs_df = dataset.replace_index(filter_probs_df, {8: 7})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config, include=[r'^v\d+$']))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, dataset.pandas["filter_probs.csv"]))
    pipe.run()

    assert_results(comp_stage.get_results())


@pytest.mark.use_cudf
@pytest.mark.parametrize("dup_index", [False, True])
def test_deserialize_multi_segment_pipe(config: Config, dataset: DatasetManager, dup_index: bool):
    """
    End to end test across mulitiple segments
    """

    filter_probs_df = dataset["filter_probs.csv"]
    if dup_index:
        filter_probs_df = dataset.replace_index(filter_probs_df, {8: 7})

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_segment_boundary(MessageMeta)
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(SerializeStage(config, include=[r'^v\d+$']))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, dataset.pandas["filter_probs.csv"]))
    pipe.run()

    assert_results(comp_stage.get_results())
