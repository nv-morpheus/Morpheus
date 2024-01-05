# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import logging
import typing

import mrc

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.module_utils import load_module

logger = logging.getLogger(__name__)


class LinearModuleSourceStage(SingleOutputSource):
    def __init__(self,
                 c: Config,
                 module_config: typing.Dict,
                 input_port_name: str,
                 output_port_name: str,
                 input_type=typing.Any,
                 output_type=typing.Any):
        super().__init__(c)

        # TODO(Devin): Fix stop requested, look at stmp source
        self._stop_requested = False
        self._input_type = input_type
        self._output_type = output_type
        self._module_config = module_config
        self._input_port_name = input_port_name
        self._output_port_name = output_port_name

    @property
    def name(self) -> str:
        return self._module_config.get("module_name", "linear_source")

    @property
    def input_count(self) -> int:
        return None

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node"""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        # Load module from the registry.
        module = load_module(self._module_config, builder=builder)

        mod_out_node = module.output_port(self._output_port_name)

        return mod_out_node
