# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import typing
from functools import wraps

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.pipeline.single_port_stage import SinglePortStage

logger = logging.getLogger(__name__)


class _DecoratedStage(SinglePortStage):

    def __init__(self, config: Config, on_data: typing.Callable):
        super().__init__(config)
        self._on_data = on_data
        self._on_data_name = f"{self._on_data.__name__}"

        signature = inspect.signature(on_data)

        try:
            first_param = next(iter(signature.parameters.values()))
            self._accept_type = first_param.annotation
            if self._accept_type is signature.empty:
                logger.warning(
                    "%s argument of %s has no type annotation, defaulting to typing.Any for the stage accept type",
                    first_param.name,
                    self._on_data_name)
                self._accept_type = typing.Any

            self._return_type = signature.return_annotation
            if self._return_type is signature.empty:
                logger.warning("Return type of %s has no type annotation, defaulting to the stage's accept type",
                               self._on_data_name)
                self._return_type = self._accept_type

        except StopIteration as e:
            raise ValueError(f"Wrapped stage functions {self._on_data_name} must have at least one parameter") from e

    @property
    def name(self) -> str:
        return f"decorated-stage-{self._on_data_name}"

    def accepted_types(self) -> typing.Tuple:
        return (self._accept_type, )

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        if self._return_type is not typing.Any:
            return_type = self._return_type
        else:
            return_type = schema.input_schema.get_type()

        schema.output_schema.set_type(return_type)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._on_data))
        builder.make_edge(input_node, node)

        return node


def stage(on_data: typing.Callable):

    @wraps(on_data)
    def wrapper(*args, **kwargs):
        return _DecoratedStage(*args, on_data=on_data, **kwargs)

    return wrapper
