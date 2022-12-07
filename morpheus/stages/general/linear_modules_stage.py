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

import logging
import typing

import srf

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.decorators import is_module_registered

logger = logging.getLogger("morpheus.{}".format(__name__))


class LinearModulesStage(SinglePortStage):
    """
    Simply loads registered module.

    The specified module will be loaded at this stage if it's registered in the module registry.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    module_config : typing.Dict
        Module configuration.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self, c: Config, module_config: typing.Dict, input_type=typing.Any, output_type=typing.Any):

        super().__init__(c)

        self._input_type = input_type
        self._ouput_type = output_type
        self._module_config = module_config

    @property
    def name(self) -> str:
        return self._module_config["module_name"]

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        return (typing.Any, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (self._input_type, )

    def _get_cpp_module_node(self, builder: srf.Builder) -> srf.SegmentObject:
        raise NotImplementedError("No C++ node is available for this module type")

    @is_module_registered
    def _load_module(self, builder: srf.Builder, module_id: str, namespace: str, module_name: str):

        module = builder.load_module(module_id, namespace, module_name, self._module_config)

        return module

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        module = self._load_module(builder=builder,
                                   module_id=self._module_config["module_id"],
                                   namespace=self._module_config["namespace"],
                                   module_name=self._module_config["module_name"])

        mod_in_stream = module.input_port("input")
        mod_out_stream = module.output_port("output")

        builder.make_edge(input_stream[0], mod_in_stream)

        return mod_out_stream, self._ouput_type
