# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import neo
import typing_utils

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.pipeline.stream_pair import StreamPair
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

    def _pre_build(self) -> typing.List[StreamPair]:
        in_ports_pairs = super()._pre_build()

        # Check the types of all inputs
        for x in in_ports_pairs:
            if (not typing_utils.issubtype(x[1], typing.Union[self.accepted_types()])):
                raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
                    self.name, x[1], self.accepted_types()))

        return in_ports_pairs

    @abstractmethod
    def _build_single(self, seg: neo.Builder, input_stream: StreamPair) -> StreamPair:
        pass

    def _build(self, seg: neo.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 1 and len(self.output_ports) == 1, \
            "SinglePortStage must have 1 input port and 1 output port"

        assert len(in_ports_streams) == 1, "Should only have 1 port on input"

        return [self._build_single(seg, in_ports_streams[0])]

    def _post_build_single(self, seg: neo.Builder, out_pair: StreamPair) -> StreamPair:
        return out_pair

    @typing.final
    def _post_build(self, seg: neo.Builder, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        ret_val = self._post_build_single(seg, out_ports_pair[0])

        logger.info("Added stage: {}\n  └─ {} -> {}".format(str(self),
                                                            pretty_print_type_name(self.input_ports[0].in_type),
                                                            pretty_print_type_name(ret_val[1])))

        return [ret_val]
