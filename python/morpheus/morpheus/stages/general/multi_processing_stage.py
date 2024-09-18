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

import functools
import inspect
import typing
from abc import abstractmethod

import mrc
import mrc.core.operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.shared_process_pool import SharedProcessPool

InputT = typing.TypeVar('InputT')
OutputT = typing.TypeVar('OutputT')


class MultiProcessingBaseStage(SinglePortStage, typing.Generic[InputT, OutputT]):

    def __init__(self, *, c: Config, process_pool_usage: float, max_in_flight_messages: int = None):
        super().__init__(c=c)

        self._process_pool_usage = process_pool_usage
        self._shared_process_pool = SharedProcessPool()
        self._shared_process_pool.wait_until_ready()

        if max_in_flight_messages is None:
            # set the multiplier to 1.5 to keep the workers busy
            self._max_in_flight_messages = int(self._shared_process_pool.total_max_workers * 1.5)
        else:
            self._max_in_flight_messages = max_in_flight_messages

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Marked as abstract to force the derived stage to provide a unique name.

        Returns:
            str: The unique name of the stage.
        """
        return "multi-processing-base-stage"

    def accepted_types(self) -> typing.Tuple:
        if hasattr(self, "__orig_class__"):
            # Derived with abstract types
            input_type = typing.get_args(self.__orig_class__)[0]  # pylint: disable=no-member

        elif hasattr(self, "__orig_bases__"):
            # Derived with concrete types
            input_type = typing.get_args(self.__orig_bases__[0])[0]  # pylint: disable=no-member

        else:
            raise RuntimeError("Could not deduct input type")

        return (input_type, )

    def compute_schema(self, schema: StageSchema):
        if hasattr(self, "__orig_class__"):
            # Derived with abstract types
            output_type = typing.get_args(self.__orig_class__)[1]  # pylint: disable=no-member

        elif hasattr(self, "__orig_bases__"):
            # Derived with concrete types
            output_type = typing.get_args(self.__orig_bases__[0])[1]

        else:
            raise RuntimeError("Could not deduct output type")

        schema.output_schema.set_type(output_type)

    def supports_cpp_node(self):
        return False

    @abstractmethod
    def _on_data(self, data: InputT) -> OutputT:
        pass

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.name, ops.map(self._on_data))
        node.launch_options.pe_count = self._max_in_flight_messages

        builder.make_edge(input_node, node)

        return node


def _get_func_signature(func: typing.Callable[[InputT], OutputT]) -> tuple[type, type]:
    signature = inspect.signature(func)

    if isinstance(func, functools.partial):
        # If the function is a partial, find the type of the first unbound argument
        bound_args = func.keywords
        input_arg = None

        for param in signature.parameters.values():
            if param.name not in bound_args:
                input_arg = param
                break

        if input_arg is None:
            raise ValueError("Could not find unbound argument in partial function")
        input_t = input_arg.annotation

    else:
        input_t = next(iter(signature.parameters.values())).annotation

    output_t = signature.return_annotation

    return (input_t, output_t)


class MultiProcessingStage(MultiProcessingBaseStage[InputT, OutputT]):

    def __init__(self,
                 *,
                 c: Config,
                 unique_name: str,
                 process_fn: typing.Callable[[InputT], OutputT],
                 process_pool_usage: float,
                 max_in_flight_messages: int = None):
        super().__init__(c=c, process_pool_usage=process_pool_usage, max_in_flight_messages=max_in_flight_messages)

        self._name = unique_name
        self._process_fn = process_fn
        self._shared_process_pool.set_usage(self.name, self._process_pool_usage)

    @property
    def name(self) -> str:
        return self._name

    def _on_data(self, data: InputT) -> OutputT:
        task = self._shared_process_pool.submit_task(self.name, self._process_fn, data)
        result = task.result()

        return result

    @staticmethod
    def create(*,
               c: Config,
               unique_name: str,
               process_fn: typing.Callable[[InputT], OutputT],
               process_pool_usage: float):

        input_t, output_t = _get_func_signature(process_fn)
        return MultiProcessingStage[input_t, output_t](c=c,
                                                       unique_name=unique_name,
                                                       process_pool_usage=process_pool_usage,
                                                       process_fn=process_fn)
