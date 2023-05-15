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

import logging
import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.module_utils import load_module

logger = logging.getLogger(__name__)


class MultiPortModulesStage(Stage):
    """
    Loads an existing, registered, MRC SegmentModule and wraps it as a Morpheus Stage.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    module_config : typing.Dict
        Module configuration.
   input_ports : typing.List[str]
        Input ports used for the registered module.
    output_ports : typing.List[str]
        Output ports used for the registered module.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_conf: typing.Dict[str, any],
                 input_ports: typing.List[str],
                 output_ports: typing.List[str],
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._module_conf = module_conf
        self._input_type = input_type
        self._ouput_type = output_type

        if not (input_ports and output_ports):
            raise ValueError("Module input and output ports must not be empty.")
        self._in_ports = input_ports
        self._out_ports = output_ports

        self._num_in_ports = len(self._in_ports)
        self._num_out_ports = len(self._out_ports)

        self._create_ports(self._num_in_ports, self._num_out_ports)

    @property
    def name(self) -> str:
        return self._module_conf.get("module_name", "multi_port_module")

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        """
        Returns input type for the current stage.
        """

        return (self._input_type, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build(self, builder: mrc.Builder, in_port_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        # Load module from the registry.
        module = load_module(self._module_conf, builder=builder)

        # Make an edges with input ports
        for index in range(self._num_in_ports):
            in_stream_node = in_port_streams[index][0]
            in_port = self._in_ports[index]
            mod_in_stream = module.input_port(in_port)
            builder.make_edge(in_stream_node, mod_in_stream)

        out_stream_pairs = []

        for index in range(self._num_out_ports):
            out_port = self._out_ports[index]
            out_stream_pairs.append((module.output_port(out_port), self._ouput_type))

        return out_stream_pairs
