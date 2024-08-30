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
from typing import Tuple

import pytest

import cudf

from _utils import assert_results
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.multi_processing_stage import MultiProcessingBaseStage
from morpheus.stages.general.multi_processing_stage import MultiProcessingStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def test_constructor(config: Config):
    stage = MultiProcessingStage.create(c=config, process_fn=lambda x: x, process_pool_usage=0.5)
    assert stage.name == "multi-processing-stage"


class DerivedMultiProcessingStage(MultiProcessingBaseStage[ControlMessage, ControlMessage]):

    def __init__(self,
                 *,
                 c: Config,
                 process_pool_usage: float,
                 add_column_name: str,
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._add_column_name = add_column_name

    @property
    def name(self) -> str:
        return "derived-multi-processing-stage"

    def accepted_types(self) -> Tuple:
        return (ControlMessage, )

    def _on_data(self, data: ControlMessage) -> ControlMessage:
        with data.payload().mutable_dataframe() as df:
            df[self._add_column_name] = "Hello"

        return data


@pytest.mark.use_python
def test_stage_pipe(config: Config, dataset_pandas: DatasetManager):

    config.num_threads = os.cpu_count()
    
    input_df = dataset_pandas["filter_probs.csv"]
    add_column_name = "new_column"
    expected_df = input_df.copy()
    expected_df[add_column_name] = "Hello"

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [cudf.DataFrame(input_df)]))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=True, message_type=ControlMessage))
    pipe.add_stage(DerivedMultiProcessingStage(c=config, process_pool_usage=0.5, add_column_name=add_column_name))
    pipe.add_stage(SerializeStage(config))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))

    pipe.run()

    assert_results(comp_stage.get_results())
