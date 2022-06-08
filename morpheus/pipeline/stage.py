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

import srf

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class Stage(_pipeline.StreamWrapper):
    """
    This class serves as the base for all pipeline stage implementations that are not source objects.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

    def _post_build(self, seg: srf.Builder, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        return out_ports_pair

    def _start(self):
        pass

    def on_start(self):
        """
        This function can be overridden to add usecase-specific implementation at the start of any stage in
        the pipeline.
        """
        pass

    def _on_complete(self, stream):

        logger.info("Stage Complete: {}".format(self.name))
