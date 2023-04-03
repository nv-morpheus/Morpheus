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
    input_port_name : str
        Name of the input port for the registered module.
    output_port_name_prefix : str
        Prefix name of the output ports for the registered module.
    num_output_ports : str
        Number of output ports for the registered module.
    input_type : default `typing.Any`
        The stage acceptable input type.
    output_type : default `typing.Any`
        The output type that the stage produces.

    """

    def __init__(self,
                 c: Config,
                 module_conf: typing.Dict[str, any],
                 input_port_name: str,
                 output_port_name_prefix: str,
                 num_output_ports: int,
                 input_type=typing.Any,
                 output_type=typing.Any):

        super().__init__(c)

        self._input_type = input_type
        self._ouput_type = output_type
        self._module_conf = module_conf
        self._input_port_name = input_port_name
        self._output_port_name_prefix = output_port_name_prefix

        if num_output_ports < 1:
            raise ValueError(f"The `output_port_count` must be >= 1, but received {num_output_ports}.")

        self._create_ports(1, num_output_ports)
        self._output_port_count = num_output_ports

    @property
    def name(self) -> str:
        return self._module_conf.get("module_name", "multi_port_module")

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

    def _build(self, builder: mrc.Builder, in_stream_pairs: typing.List[StreamPair]) -> typing.List[StreamPair]:

        in_ports_len = len(in_stream_pairs)
        if in_ports_len != 1:
            raise ValueError(f"Only 1 input is supported, but recieved {in_ports_len}.")

        in_stream_node = in_stream_pairs[0][0]

        # Load module from the registry.
        module = load_module(self._module_conf, builder=builder)
        mod_in_stream = module.input_port(self._input_port_name)

        builder.make_edge(in_stream_node, mod_in_stream)

        out_stream_pairs = []

        count = 0
        while (count < self._output_port_count):
            out_port = f"{self._output_port_name_prefix}-{count}"
            out_stream_pairs.append((module.output_port(out_port), self._ouput_type))
            count += 1

        return out_stream_pairs
