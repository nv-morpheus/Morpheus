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
from abc import abstractmethod

import mrc

import morpheus.pipeline as _pipeline

logger = logging.getLogger(__name__)


class Stage(_pipeline.StreamWrapper):
    """
    This class serves as the base for all pipeline stage implementations that are not source objects.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def _post_build(self, builder: mrc.Builder, out_ports_nodes: list[mrc.SegmentObject]) -> list[mrc.SegmentObject]:

        return out_ports_nodes

    def _start(self):
        pass

    @abstractmethod
    def output_types(self, parent_output_types: list[type]) -> list[type]:
        """
        Return the output types for this stage based on the incoming types of parent stages. Derived classes should
        override this method, if the `parent_output_types` are incompatible then the stage should rase a
        `RuntimeError` exception.

        Returns
        -------
        list
            Output types.

        """
        pass

    def on_start(self):
        """
        This function can be overridden to add usecase-specific implementation at the start of any stage in
        the pipeline.
        """
        pass

    async def start_async(self):
        """
        This function is called along with on_start during stage initialization. Allows stages to utilize the
        asyncio loop if needed.
        """
        pass

    def _on_complete(self, stream):  # pylint: disable=unused-argument
        logger.info("Stage Complete: %s", self.name)
