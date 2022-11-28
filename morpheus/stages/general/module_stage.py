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

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.modules.module_factory import ModuleFactory

logger = logging.getLogger("morpheus.{}".format(__name__))


def get_type_class(type_class):
    module, classname = type_class.rsplit('.', 1)
    # load the type module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module)
    # get the type class, will raise AttributeError if class cannot be found
    type_class = getattr(module, classname)

    return type_class


class ModuleStage(SinglePortStage):

    def __init__(self, c: Config, mc: typing.Dict):

        super().__init__(c)

        self._mc = mc
        self._module_id = mc["module_id"]
        self._module_name = mc["module_name"]
        self._module_ns = mc["module_namespace"]

        self._input_type_class = get_type_class(mc["input_type_class"])
        self._output_type_class = get_type_class(mc["output_type_class"])

        self._registry = srf.ModuleRegistry()

    @property
    def name(self) -> str:
        return self._module_name

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        return (self._input_type_class, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (self._input_type_class, )

    def _get_cpp_module_node(self, builder: srf.Builder) -> srf.SegmentObject:
        raise NotImplementedError("No C++ node is available for this module type")

    def _register_modules(self):

        # Register if there are any inner modules.
        if "modules" in self._mc and self._mc["modules"] is not None:

            inner_mc = self._mc["modules"]

            for key in inner_mc.keys():
                module_conf = inner_mc[key]
                unique_name = self.unique_name + "-" + module_conf["module_id"]
                module_conf["unique_name"] = unique_name

                ModuleFactory.register_module(self._config, module_conf)

        self._mc["unique_name"] = self.unique_name

        # Register a module
        ModuleFactory.register_module(self._config, self._mc)

        # Verify if module exists in the namespace.
        if not self._registry.contains(self._module_id, self._module_ns):
            raise Exception("Module: {} with Namespace: {} doesn't exists in the registry".format(
                self._module_id, self._module_ns))

        logger.debug("Available modules: {}".format(self._registry.registered_modules()))

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        self._register_modules()

        # Load registered module
        module = builder.load_module(self._module_id, self._module_ns, self._module_name, self._mc)

        mod_in_stream = module.input_port("input")
        mod_out_stream = module.output_port("output")

        builder.make_edge(input_stream[0], mod_in_stream)

        return mod_out_stream, self._output_type_class
