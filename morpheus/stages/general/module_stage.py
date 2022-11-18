# Copyright (c) 2022, NVIDIA CORPORATION.
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

import importlib
import logging
import typing

import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger("morpheus.{}".format(__name__))


class ModuleStage(SinglePortStage):

    def __init__(
        self,
        c: Config,
        module_id: str,
        module_namespace: str,
        module_name: str,
        input_type_class: str,
        output_type_class: str,
        module_config: typing.Dict = {},
    ):

        super().__init__(c)

        self._registry = srf.ModuleRegistry
        self._module_id = module_id
        self._module_namespace = module_namespace
        self._module_name = module_name
        self._module_config = module_config

        self._input_type_class = self.get_type_class(input_type_class)
        self._output_type_class = self.get_type_class(output_type_class)

    def get_type_class(self, type_class):
        module, classname = self._input_type_class.rsplit('.', 1)
        # load the type module, will raise ImportError if module cannot be loaded
        module = importlib.import_module(module)
        # get the type class, will raise AttributeError if class cannot be found
        type_class = getattr(module, classname)

        return type_class

    @property
    def name(self) -> str:
        return self._module_name

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        return (self._input_type_class, )
    
    @register_module
    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        if not self._registry.contains(self._module_id, self._module_namespace):
            raise Exception("Module: {} with Namespace: {} doesn't exists in the registry".format(
                self._module_id, self._module_namespace))

        module = builder.load_module(self._module_id, self._module_namespace, self._module_name, self._module_config)

        mod_in_stream = module.input_port("input")
        mod_out_stream = module.output_port("output")

        builder.make_edge(input_stream[0], mod_in_stream)

        return mod_out_stream, self._output_type_class
