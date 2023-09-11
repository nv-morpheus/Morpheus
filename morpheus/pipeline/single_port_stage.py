# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
from abc import abstractmethod

import mrc
import typing_utils

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)


class SinglePortStage(_pipeline.Stage):
    """
    Class used for building stages with single input port and single output port.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._create_ports(1, 1)

    @abstractmethod
    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned. Derived classes should override this method. An
        error will be generated if the input types to the stage do not match one of the available types
        returned from this method.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        pass

    def output_types(self, parent_output_types: list[type]) -> list[type]:
        """
        Return the output type for this stage.

        Returns
        -------
        list
            Output types.

        """
        accepted_types = typing.Union[self.accepted_types()]

        assert len(parent_output_types) == 1, "SinglePortStage must have 1 input port and 1 output port"
        parent_output_type = parent_output_types[0]
        if (not typing_utils.issubtype(parent_output_type, accepted_types)):
            raise RuntimeError((f"The {self.name} stage cannot handle input of {parent_output_type}. "
                                f"Accepted input types: {self.accepted_types()}"))

        return [self.output_type(parent_output_type)]

    @abstractmethod
    def output_type(self, parent_output_type: type) -> type:
        """
        Return the output type for this stage. Derived classes should override this method.

        Returns
        -------
        type
            Output type.

        """
        pass

    @abstractmethod
    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        pass

    def _build(self, builder: mrc.Builder, input_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 1 and len(self.output_ports) == 1, \
            "SinglePortStage must have 1 input port and 1 output port"

        assert len(input_nodes) == 1, "Should only have 1 input node"

        return [self._build_single(builder, input_nodes[0])]

    def _post_build_single(self, _: mrc.Builder, out_node: mrc.SegmentObject) -> mrc.SegmentObject:
        return out_node

    @typing.final
    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        ret_val = self._post_build_single(builder, out_ports_nodes[0])

        # pylint: disable=logging-format-interpolation
        logger.info("Added stage: %s\n  └─ %s -> %s",
                    str(self),
                    pretty_print_type_name(self.input_ports[0].in_type),
                    pretty_print_type_name(self.output_ports[0].out_type))

        return [ret_val]
