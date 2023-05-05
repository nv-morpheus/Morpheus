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
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.module_utils import load_module

logger = logging.getLogger(__name__)


class MultiInputportModulesStage(Stage):
    """
    Wraps an existing MRC SegmentModule that has multiple input ports and a single output port,
    and converts it into a Morpheus stage.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    module_conf : typing.Dict
        Module configuration.
    num_input_ports_to_merge : str
        Number of nodes stream data to be combined.
    input_port_name_prefix : str
        Registered module's input ports' name prefix.
    output_port_name: int
        Name of the registered module's output port.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_conf: typing.Dict[str, any],
                 num_input_ports_to_merge: int,
                 input_port_name_prefix: str = "input",
                 output_port_name: str = "output",
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._input_type = input_type
        self._ouput_type = output_type
        self._module_conf = module_conf
        self._input_port_name_prefix = input_port_name_prefix
        self._output_port_name = output_port_name

        if num_input_ports_to_merge < 1:
            raise ValueError(f"The `num_input_ports_to_merge` must be >= 1, but received {num_input_ports_to_merge}.")

        self._create_ports(num_input_ports_to_merge, 1)
        self._num_input_ports_to_merge = num_input_ports_to_merge

    @property
    def name(self) -> str:
        return self._module_conf.get("module_name", "multi_inputport_module")

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        """
        Returns input type for the current stage.
        """

        return (typing.Any, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build(self, builder: mrc.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:

        # Load multiplexer module from the registry.
        module = load_module(self._module_conf, builder=builder)

        for index in range(self._num_input_ports_to_merge):
            in_stream_node = in_ports_streams[index][0]
            input_port_name = f"{self._input_port_name_prefix}-{index}"
            mod_in_stream = module.input_port(input_port_name)
            builder.make_edge(in_stream_node, mod_in_stream)

        return [(module.output_port(self._output_port_name), self._ouput_type)]
