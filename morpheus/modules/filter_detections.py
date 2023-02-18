# Copyright (c) 2023, NVIDIA CORPORATION.
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
import pickle
import typing

import cupy as cp
import mrc
import numpy as np
import typing_utils
from mrc.core import operators as ops

from morpheus.common import FilterSource
from morpheus.messages import MultiMessage
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.utils.module_ids import FILTER_DETECTIONS
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(FILTER_DETECTIONS, MODULE_NAMESPACE)
def filter_detections(builder: mrc.Builder):
    """
    Filter message by a classification threshold.

    The FilterDetections is used to filter rows from a dataframe based on values in a tensor using a specified
    criteria. Rows in the `meta` dataframe are excluded if their associated value in the `probs` array is less than or
    equal to `threshold`.

    This module can operate in two different modes set by the `copy` argument.
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
    builder : mrc.Builder
        mrc Builder object.
    """

    config = get_module_config(FILTER_DETECTIONS, builder)

    field_name = config.get("field_name", "probs")
    threshold = config.get("threshold", 0.5)
    filter_source = config.get("filter_source", "AUTO")
    copy = config.get("copy", True)

    schema_config = config.get("schema", None)
    input_message_type = schema_config.get("input_message_type", None)
    encoding = schema_config.get("encoding", None)

    message_type = pickle.loads(bytes(input_message_type, encoding))

    def find_detections(x: MultiMessage, filter_source) -> typing.Union[cp.ndarray, np.ndarray]:

        # Determind the filter source
        if filter_source == FilterSource.TENSOR:
            filter_source = x.get_output(field_name)
        else:
            filter_source = x.get_meta(field_name).values

        if (isinstance(filter_source, np.ndarray)):
            array_mod = np
        else:
            array_mod = cp

        # Get per row detections
        detections = (filter_source > threshold)

        if (len(detections.shape) > 1):
            detections = detections.any(axis=1)

        # Surround in False to ensure we get an even number of pairs
        detections = array_mod.concatenate([array_mod.array([False]), detections, array_mod.array([False])])

        return array_mod.where(detections[1:] != detections[:-1])[0].reshape((-1, 2))

    def filter_copy(x: MultiMessage) -> MultiMessage:
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

        true_pairs = find_detections(x, filter_source)

        return x.copy_ranges(true_pairs)

    def filter_slice(x: MultiMessage) -> typing.List[MultiMessage]:
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
            true_pairs = find_detections(x, filter_source)
            for pair in true_pairs:
                pair = tuple(pair.tolist())
                if ((pair[1] - pair[0]) > 0):
                    output_list.append(x.get_slice(*pair))

        return output_list

    if filter_source == "AUTO":
        if (typing_utils.issubtype(message_type, MultiResponseMessage)):
            filter_source = FilterSource.TENSOR
        else:
            filter_source = FilterSource.DATAFRAME

        logger.debug(f"filter_source was set to Auto, infering a filter source of {filter_source} based on an input "
                     "message type of {message_type}")
    elif filter_source == "DATAFRAME":
        filter_source = FilterSource.DATAFRAME
    else:
        raise Exception("Unknown filter source: {}".format(filter_source))

    if copy:
        node = builder.make_node(FILTER_DETECTIONS, filter_copy)
    else:
        # Convert list back to individual messages
        def flatten_fn(obs: mrc.Observable, sub: mrc.Subscriber):
            obs.pipe(ops.map(filter_slice), ops.flatten()).subscribe(sub)

        node = builder.make_node_full(FILTER_DETECTIONS, flatten_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
