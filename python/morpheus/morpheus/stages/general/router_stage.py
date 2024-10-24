# Copyright (c) 2024, NVIDIA CORPORATION.
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
import mrc.core.segment
import typing_utils

import morpheus.pipeline as _pipeline  # pylint: disable=cyclic-import
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin

logger = logging.getLogger(__name__)


@register_stage("router")
class RouterStage(GpuAndCpuMixin, PassThruTypeMixin, _pipeline.Stage):
    """
    Buffer results.

    The input messages are buffered by this stage class for faster access to downstream stages. Allows
    upstream stages to run faster than downstream stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self,
                 c: Config,
                 *,
                 keys: list[str],
                 key_fn: typing.Callable[[ControlMessage], str],
                 is_runnable: bool = False) -> None:
        super().__init__(c)

        self._keys = keys
        self._key_fn = key_fn
        self._is_runnable = is_runnable

        self._router: mrc.core.segment.SegmentObject | None = None

        self._create_ports(1, len(keys))

    @property
    def name(self) -> str:
        return "router"

    def supports_cpp_node(self):
        return True

    def _pre_compute_schema(self, schema: _pipeline.StageSchema):
        # Pre-flight check to verify that the input type is one of the accepted types
        super()._pre_compute_schema(schema)
        input_type = schema.input_type
        if (not typing_utils.issubtype(input_type, ControlMessage)):
            raise RuntimeError((f"The {self.name} stage cannot handle input of {input_type}. "
                                f"Accepted input types: {(ControlMessage,)}"))

    def compute_schema(self, schema: _pipeline.StageSchema):

        # Get the input type
        input_type = schema.input_type

        for port_idx in range(len(self._keys)):
            schema.output_schemas[port_idx].set_type(input_type)

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        def _key_fn_wrapper(msg) -> str:
            return self._key_fn(msg)

        assert len(input_nodes) == 1, "Router stage should have exactly one input node"

        if (self._build_cpp_node()):
            import morpheus._lib.stages as _stages

            if (self._is_runnable):
                self._router = _stages.RouterControlMessageRunnableStage(builder,
                                                                         self.unique_name,
                                                                         router_keys=self._keys,
                                                                         key_fn=_key_fn_wrapper)
            else:
                self._router = _stages.RouterControlMessageComponentStage(builder,
                                                                          self.unique_name,
                                                                          router_keys=self._keys,
                                                                          key_fn=_key_fn_wrapper)
        else:
            from mrc.core.node import Router
            from mrc.core.node import RouterComponent

            if (self._is_runnable):
                self._router = Router(builder, self.unique_name, router_keys=self._keys, key_fn=_key_fn_wrapper)
            else:
                self._router = RouterComponent(builder,
                                               self.unique_name,
                                               router_keys=self._keys,
                                               key_fn=_key_fn_wrapper)

        if (self._is_runnable):
            self._router.launch_options.engines_per_pe = 10

        builder.make_edge(input_nodes[0], self._router)

        return [self._router.get_child(k) for k in self._keys]
