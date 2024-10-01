# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import typing
from functools import partial
from typing import Generator

import pandas as pd
import pytest

import cudf

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.multi_processing_stage import MultiProcessingBaseStage
from morpheus.stages.general.multi_processing_stage import MultiProcessingStage
from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def _create_df(count: int) -> pd.DataFrame:
    return pd.DataFrame({"a": range(count)}, {"b": range(count)})


def _process_df(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    df[column] = value
    return df


def test_create_stage_type_deduction(config: Config, dataset_pandas: DatasetManager):

    # Test create() with normal function
    mp_stage = MultiProcessingStage.create(c=config,
                                           unique_name="multi-processing-stage-1",
                                           process_fn=_create_df,
                                           process_pool_usage=0.1)
    assert mp_stage.name == "multi-processing-stage-1"
    input_t, output_t = typing.get_args(mp_stage.__orig_class__)  # pylint: disable=no-member
    assert input_t == int
    assert output_t == pd.DataFrame

    # Test create() with partial function with 1 unbound argument
    df = dataset_pandas["csv_sample.csv"]
    partial_fn = partial(_process_df, df=df, value="new_value")

    mp_stage = MultiProcessingStage.create(c=config,
                                           unique_name="multi-processing-stage-2",
                                           process_fn=partial_fn,
                                           process_pool_usage=0.1)

    assert mp_stage.name == "multi-processing-stage-2"
    input_t, output_t = typing.get_args(mp_stage.__orig_class__)  # pylint: disable=no-member
    assert mp_stage.accepted_types() == (str, )
    assert input_t == str
    assert output_t == pd.DataFrame

    # Invalid case: create() with partial function with 0 unbound argument
    invalid_partial_fn = partial(_process_df, df=df, column="new_column", value="new_value")
    with pytest.raises(ValueError):
        MultiProcessingStage.create(c=config,
                                    unique_name="multi-processing-stage-3",
                                    process_fn=invalid_partial_fn,
                                    process_pool_usage=0.1)

    # Invalid case: create() with function with more than 1 arguments
    invalid_partial_fn = partial(_process_df, df=df)
    with pytest.raises(ValueError):
        MultiProcessingStage.create(c=config,
                                    unique_name="multi-processing-stage-4",
                                    process_fn=invalid_partial_fn,
                                    process_pool_usage=0.1)


class DerivedMultiProcessingStage(MultiProcessingBaseStage[ControlMessage, ControlMessage]):

    def __init__(self,
                 *,
                 c: Config,
                 process_pool_usage: float,
                 add_column_name: str,
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._add_column_name = add_column_name
        self._shared_process_pool.set_usage(self.name, self._process_pool_usage)

    @property
    def name(self) -> str:
        return "derived-multi-processing-stage"

    def _on_data(self, data: ControlMessage) -> ControlMessage:

        input_df = data.payload().copy_dataframe()
        pdf = input_df.to_pandas()
        partial_process_fn = partial(_process_df, column=self._add_column_name, value="Hello")

        task = self._shared_process_pool.submit_task(self.name, partial_process_fn, pdf)

        df = cudf.DataFrame.from_pandas(task.result())
        meta = MessageMeta(df)
        data.payload(meta)

        return data


def test_derived_stage_type_deduction(config: Config):

    mp_stage = DerivedMultiProcessingStage(c=config, process_pool_usage=0.1, add_column_name="new_column")
    assert mp_stage.name == "derived-multi-processing-stage"
    assert mp_stage.accepted_types() == (ControlMessage, )

    input_t, output_t = typing.get_args(mp_stage.__orig_bases__[0])  # pylint: disable=no-member
    assert input_t == ControlMessage
    assert output_t == ControlMessage


def pandas_dataframe_generator(dataset_pandas: DatasetManager, count: int) -> Generator[pd.DataFrame, None, None]:

    df = dataset_pandas["csv_sample.csv"]
    for _ in range(count):
        yield df


def test_created_stage_pipe(config: Config, dataset_pandas: DatasetManager):

    config.num_threads = os.cpu_count()

    input_df = dataset_pandas["csv_sample.csv"]

    expected_df = input_df.copy()
    expected_df["new_column"] = "Hello"

    df_count = 100
    df_generator = partial(pandas_dataframe_generator, dataset_pandas, df_count)

    partial_fn = partial(_process_df, column="new_column", value="Hello")

    pipe = LinearPipeline(config)
    pipe.set_source(InMemoryDataGenStage(config, df_generator, output_data_type=pd.DataFrame))
    pipe.add_stage(MultiProcessingStage[pd.DataFrame, pd.DataFrame].create(c=config,
                                                                           unique_name="multi-processing-stage-5",
                                                                           process_fn=partial_fn,
                                                                           process_pool_usage=0.1))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    for df in sink_stage.get_messages():
        assert df.equals(expected_df)


def test_derived_stage_pipe(config: Config, dataset_pandas: DatasetManager):

    config.num_threads = os.cpu_count()

    input_df = dataset_pandas["csv_sample.csv"]
    add_column_name = "new_column"
    expected_df = input_df.copy()
    expected_df[add_column_name] = "Hello"

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=True))
    pipe.add_stage(DerivedMultiProcessingStage(c=config, process_pool_usage=0.1, add_column_name=add_column_name))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    pipe.run()

    assert_results(comp_stage.get_results())


def test_multiple_stages_pipe(config: Config, dataset_pandas: DatasetManager):
    config.num_threads = os.cpu_count()

    input_df = dataset_pandas["csv_sample.csv"]

    expected_df = input_df.copy()
    expected_df["new_column_1"] = "new_value"
    expected_df["new_column_2"] = "Hello"

    df_count = 100
    df_generator = partial(pandas_dataframe_generator, dataset_pandas, df_count)

    partial_fn = partial(_process_df, column="new_column_1", value="new_value")

    @stage
    def pdf_to_control_message_stage(pdf: pd.DataFrame) -> ControlMessage:
        df = cudf.DataFrame.from_pandas(pdf)
        meta = MessageMeta(df)
        msg = ControlMessage()
        msg.payload(meta)

        return msg

    pipe = LinearPipeline(config)
    pipe.set_source(InMemoryDataGenStage(config, df_generator, output_data_type=pd.DataFrame))
    pipe.add_stage(
        MultiProcessingStage.create(c=config,
                                    unique_name="multi-processing-stage-6",
                                    process_fn=partial_fn,
                                    process_pool_usage=0.1))
    pipe.add_stage(pdf_to_control_message_stage(config))
    pipe.add_stage(DerivedMultiProcessingStage(c=config, process_pool_usage=0.1, add_column_name="new_column_2"))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    pipe.run()

    assert_results(comp_stage.get_results())
