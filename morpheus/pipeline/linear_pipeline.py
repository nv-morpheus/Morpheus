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

import morpheus.pipeline as _pipeline
from morpheus.config import Config

logger = logging.getLogger(__name__)


class LinearPipeline(_pipeline.Pipeline):
    """
    This class is used to build linear pipelines where we have a single output source stage followed by stages that are
    executed sequentially in the order they were added.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._linear_stages: typing.List[_pipeline.StreamWrapper] = []

    def set_source(self, source: _pipeline.SourceStage):
        """
        Set a pipeline's source stage to consume messages before it begins executing stages. This must be
        called once before `build_and_start`.

        Parameters
        ----------
        source : `SourceStage`
            The source stage wraps the implementation in a stream that allows it to read from Kafka or a file.

        """

        if (len(self._sources) > 0 and source not in self._sources):
            logger.warning(
                "LinearPipeline already has a source. Setting a new source will clear out all existing stages")

            self._sources.clear()

        # Store the source in sources
        self._sources.add(source)

        if (len(self._linear_stages) > 0):
            logger.warning("Clearing %d stages from pipeline", len(self._linear_stages))
            self._linear_stages.clear()

        # Need to store the source in the pipeline
        super().add_node(source)

        # Store this as the first one in the linear stages. Must be index 0
        self._linear_stages.append(source)

    def add_stage(self, stage: _pipeline.SinglePortStage):
        """
        Add stages to the pipeline. All `Stage` classes added with this method will be executed sequentially
        inthe order they were added.

        Parameters
        ----------
        stage : `Stage`
            The stage object to add. It cannot be already added to another `Pipeline` object.

        """

        assert len(self._linear_stages) > 0, "A source must be set on a LinearPipeline before adding any stages"
        assert isinstance(stage, _pipeline.SinglePortStage), "Only `SinglePortStage` stages are accepted in `add_stage()`"

        # Make an edge between the last node and this one
        self.add_edge(self._linear_stages[-1], stage)

        self._linear_stages.append(stage)
