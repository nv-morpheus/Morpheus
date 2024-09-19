# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import functools
import logging
import types

import mrc
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus_llm.llm import LLMEngine

logger = logging.getLogger(__name__)


class LLMEngineStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    Stage for executing an LLM engine within a Morpheus pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    engine : `morpheus_llm.llm.LLMEngine`
        LLM engine instance to execute.
   """

    def __init__(self, c: Config, *, engine: LLMEngine):
        super().__init__(c)

        self._engine = engine

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "llm-engine"

    def accepted_types(self) -> tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        tuple(`ControlMessage`, )
            Accepted input types.

        """
        return (ControlMessage, )

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node."""
        return True

    def _cast_control_message(self, message: ControlMessage, *, cpp_messages_lib: types.ModuleType) -> ControlMessage:
        """
        LLMEngineStage does not contain a Python implementation, however it is capable of running in cpu-only mode.
        This method is needed to cast the Python ControlMessage to a C++ ControlMessage.

        This is different than casting from the Python bindings for the C++ ControlMessage to a C++ ControlMessage.
        """
        return cpp_messages_lib.ControlMessage(message, no_cast=True)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        import morpheus_llm._lib.llm as _llm
        node = _llm.LLMEngineStage(builder, self.unique_name, self._engine)
        node.launch_options.pe_count = 1

        if self._config.execution_mode == ExecutionMode.CPU:
            import morpheus._lib.messages as _messages
            cast_fn = functools.partial(self._cast_control_message, cpp_messages_lib=_messages)
            pre_node = builder.make_node(f"{self.unique_name}-pre-cast", ops.map(cast_fn))
            builder.make_edge(input_node, pre_node)

            input_node = pre_node

        builder.make_edge(input_node, node)

        return node
