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

import cupy as cp
import srf

import morpheus._lib.stages as _stages
from morpheus.config import Config
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class FilterDetectionsStage(SinglePortStage):
    """
    This Stage class is used to filter results based on a given criteria.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    threshold : float
        Threshold to classify, default is 0.5.

    """

    def __init__(self, c: Config, threshold: float = 0.5):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "filter"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiResponseProbsMessage`, ]
            Accepted input types.

        """
        return (MultiResponseProbsMessage, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def filter(self, x: MultiResponseProbsMessage) -> MultiResponseProbsMessage:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiResponseProbsMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        `morpheus.pipeline.messages.MultiResponseProbsMessage`
            List of filtered messages.

        """
        # Get per row detections
        detections = (x.probs > self._threshold).any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = cp.concatenate([cp.array([False]), detections, cp.array([False])])

        true_pairs = cp.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))
        return x.copy_ranges(true_pairs)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        if self._build_cpp_node():
            stream = _stages.FilterDetectionsStage(builder, self.unique_name, self._threshold)
        else:
            stream = builder.make_node(self.unique_name, self.filter)

        builder.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage
