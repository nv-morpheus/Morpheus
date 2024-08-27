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

import typing
from abc import ABC
from abc import abstractmethod

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.shared_process_pool import SharedProcessPool

InputT = typing.TypeVar('InputT')
OutputT = typing.TypeVar('OutputT')


class MultiProcessingBaseStage(SinglePortStage, ABC, typing.Generic[InputT, OutputT]):

    def __init__(self, *, c: Config, process_pool_usage: float, max_in_flight_messages: int = None):
        super().__init__(c=c)

        self._process_pool_usage = process_pool_usage
        self._shared_process_pool = SharedProcessPool()
        self._shared_process_pool.set_usage(self.name, self._process_pool_usage)

        if max_in_flight_messages is None:
            # set the multiplier to 1.5 to keep the workers busy
            self._max_in_flight_messages = int(self._shared_process_pool.total_max_workers * 1.5)
        else:
            self._max_in_flight_messages = max_in_flight_messages

        # self._max_in_flight_messages = 1

    @property
    def name(self) -> str:
        return "multi-processing-base-stage"

    def accepted_types(self) -> typing.Tuple:
        return (InputT, )

    def compute_schema(self, schema: StageSchema):
        for (port_idx, port_schema) in enumerate(schema.input_schemas):
            schema.output_schemas[port_idx].set_type(port_schema.get_type())

    @abstractmethod
    def _on_data(self, data: InputT) -> OutputT:
        pass

    def supports_cpp_node(self):
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.name, ops.map(self._on_data))
        node.launch_options.pe_count = self._max_in_flight_messages

        builder.make_edge(input_node, node)

        return node


class MultiProcessingStage(MultiProcessingBaseStage[InputT, OutputT]):

    def __init__(self,
                 *,
                 c: Config,
                 process_pool_usage: float,
                 process_fn: typing.Callable[[InputT], OutputT],
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._process_fn = process_fn

    @property
    def name(self) -> str:
        return "multi-processing-stage"

    def _on_data(self, data: InputT) -> OutputT:

        future = self._shared_process_pool.submit_task(self.name, self._process_fn, data)
        result = future.result()

        return result

    @staticmethod
    def create(*, c: Config, process_fn: typing.Callable[[InputT], OutputT], process_pool_usage: float):

        return MultiProcessingStage[InputT, OutputT](c=c, process_pool_usage=process_pool_usage, process_fn=process_fn)
