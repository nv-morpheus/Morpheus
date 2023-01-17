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

import cupy as cp
import mrc
import numpy as np
from mrc.core import operators as ops

import morpheus._lib.stages as _stages
from morpheus._lib.common import FilterSource
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseProbsMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("filter")
class FilterDetectionsStage(SinglePortStage):
    """
    Filter message by a classification threshold.

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
    Note: In most other stages, messages emitted contain a reference to the original `MessageMeta` emitted into the
    pipeline by the source stage. When using copy mode this won't be the case and could cause the original `MessageMeta`
    to be deallocated after this stage.

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
    filter_source : `from morpheus._lib.common.FilterSource`, default = 'auto'
        Indicate if we are operating on is an output tensor or a field in the DataFrame.
        Choosing `Auto` will default to `TENSOR` and in a future release will change to `DATAFRAME`
    field_name : str
        Name of the tensor or DataFrame column to use as the filter criteria
    """

    def __init__(self,
                 c: Config,
                 threshold: float = 0.5,
                 copy: bool = True,
                 filter_source: FilterSource = FilterSource.Auto,
                 field_name: str = "probs"):
        super().__init__(c)

        # Probability to consider a detection
        self._threshold = threshold
        self._copy = copy

        if filter_source == FilterSource.Auto:
            filter_source = FilterSource.TENSOR

        self._filter_source = filter_source
        self._field_name = field_name

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

        if self._filter_source == FilterSource.TENSOR:
            return (MultiResponseProbsMessage, )
        else:
            return (MultiMessage, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def _find_detections(self, x: MultiMessage) -> typing.Union[cp.ndarray, np.ndarray]:
        # Determind the filter source
        if self._filter_source == FilterSource.TENSOR:
            filter_source = x.get_output(self._field_name)
        else:
            filter_source = x.get_meta(self._field_name).values

        if (isinstance(filter_source, np.ndarray)):
            array_mod = np
        else:
            array_mod = cp

        # Get per row detections
        detections = (filter_source > self._threshold)

        if (len(detections.shape) > 1):
            detections = detections.any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = array_mod.concatenate([array_mod.array([False]), detections, array_mod.array([False])])

        return array_mod.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

    def filter_copy(self, x: MultiMessage) -> MultiMessage:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        `morpheus.pipeline.messages.MultiMessage`
            A new message containing a copy of the rows above the threshold.

        """
        if x is None:
            return None

        true_pairs = self._find_detections(x)
        return x.copy_ranges(true_pairs)

    def filter_slice(self, x: MultiMessage) -> typing.List[MultiMessage]:
        """
        This function uses a threshold value to filter the messages.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiMessage`
            Response message with probabilities calculated from inference results.

        Returns
        -------
        typing.List[`morpheus.pipeline.messages.MultiMessage`]
            List of filtered messages.

        """
        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        output_list = []
        if x is not None:
            true_pairs = self._find_detections(x)
            for pair in true_pairs:
                pair = tuple(pair.tolist())
                if ((pair[1] - pair[0]) > 0):
                    output_list.append(x.get_slice(*pair))

        return output_list

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        if self._build_cpp_node():
            stream = _stages.FilterDetectionsStage(builder,
                                                   self.unique_name,
                                                   self._threshold,
                                                   self._copy,
                                                   self._filter_source,
                                                   self._field_name)
        else:
            if self._copy:
                stream = builder.make_node(self.unique_name, self.filter_copy)
            else:
                # Convert list back to individual messages
                def flatten_fn(obs: mrc.Observable, sub: mrc.Subscriber):
                    obs.pipe(ops.map(self.filter_slice), ops.flatten()).subscribe(sub)

                stream = builder.make_node_full(self.unique_name, flatten_fn)

        builder.make_edge(input_stream[0], stream)

        return stream, input_stream[1]
