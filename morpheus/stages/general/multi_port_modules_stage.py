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
   input_ports_key : str
        Key within the module configuration dictionary that contains the input ports information for the module.
    output_ports_key : str
        key within the module configuration dictionary that contains the output ports information for the module.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_conf: typing.Dict[str, any],
                 input_ports_key: str,
                 output_ports_key: str,
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._module_conf = module_conf
        self._input_type = input_type
        self._ouput_type = output_type

        keys_to_validate = [input_ports_key, output_ports_key]

        if all(key in module_conf and module_conf[key] for key in keys_to_validate):
            input_ports_value = module_conf.get(input_ports_key)
            output_ports_value = module_conf.get(output_ports_key)

            if isinstance(input_ports_value, str):
                input_ports_value = [input_ports_value]
            if isinstance(output_ports_value, str):
                output_ports_value = [output_ports_value]

            self._in_ports = input_ports_value
            self._out_ports = output_ports_value
        else:
            raise ValueError(
                "Provided `input_ports_key` or `output_ports_key` either not available or empty in the module configuration."
            )

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
