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
from srf.core import operators as ops

import morpheus._lib.stages as _stages
from morpheus.config import Config
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class FilterDetectionsStage(SinglePortStage):
    """
    The FilterDetectionsStage is used to filter rows from a dataframe based on values in a tensor using a specified
    criteria. Rows in the `meta` dataframe are excluded if their associated value in the `probs` array is less than or
    equal to `threshold`.

    This stage can operate in two different modes set by the `copy` argument.
    When the `copy` argument is `True` (default), rows that meet the filter criteria are copied into a new dataframe.
    When `False` sliced views are used instead.

    Setting `copy=True` should be used when the number of matching records is expected to be both high and in
    non-adjacent rows. In this mode, the stage will generate only one output message for each incoming message,
    regardless of the size of the input and the number of matching records. However this comes at the cost of needing to
    allocate additional memory and perform the copy.

    Setting `copy=False` should be used when either the number of matching records is expected to be very low or are
    likely to be contained in adjacent rows. In this mode, slices of contiguous blocks of rows are emitted in multiple
    output messages. Performing a slice is relatively low-cost, however for each incoming message the number of emitted
    messages could be high (in the worst case scenario as high as half the number of records in the incoming message).
    Depending on the downstream stages, this can cause performance issues, especially if those stages need to acquire
    the Python GIL.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    threshold : float
        Threshold to classify, default is 0.5.

    copy : bool
        Whether or not to perform a copy.
    """

    def __init__(self, c: Config, threshold: float = 0.5, copy: bool = True):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold
        self._copy = copy

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

    def _find_detections(self, x: MultiResponseProbsMessage) -> cp.ndarray:
        # Get per row detections
        detections = (x.probs > self._threshold).any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = cp.concatenate([cp.array([False]), detections, cp.array([False])])

        return cp.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

    def filter_copy(self, x: MultiResponseProbsMessage) -> MultiResponseProbsMessage:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiResponseProbsMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        `morpheus.pipeline.messages.MultiResponseProbsMessage`
            A new message containing a copy of the rows above the threshold.

        """
        true_pairs = self._find_detections(x)
        return x.copy_ranges(true_pairs)

    def filter_slice(self, x: MultiResponseProbsMessage) -> typing.List[MultiResponseProbsMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiResponseProbsMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        typing.List[`morpheus.pipeline.messages.MultiResponseProbsMessage`]
            List of filtered messages.

        """
        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        output_list = []

        true_pairs = self._find_detections(x)
        for pair in true_pairs:
            pair = tuple(pair.tolist())
            mess_offset = x.mess_offset + pair[0]
            mess_count = pair[1] - pair[0]

            # Filter empty message groups
            if (mess_count == 0):
                continue

            output_list.append(
                MultiResponseProbsMessage(x.meta,
                                          mess_offset=mess_offset,
                                          mess_count=mess_count,
                                          memory=x.memory,
                                          offset=pair[0],
                                          count=mess_count))

        return output_list

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        if self._build_cpp_node():
            stream = _stages.FilterDetectionsStage(builder, self.unique_name, self._threshold, self._copy)
        else:
            if self._copy:
                stream = builder.make_node(self.unique_name, self.filter_copy)
            else:
                # Convert list back to individual MultiResponseProbsMessage
                def flatten_fn(obs: srf.Observable, sub: srf.Subscriber):
                    obs.pipe(ops.map(self.filter_slice), ops.flatten()).subscribe(sub)

                stream = builder.make_node_full(self.unique_name, flatten_fn)

        builder.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage
