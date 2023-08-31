# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import numpy as np
import typing_utils

from morpheus.common import FilterSource
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage

logger = logging.getLogger(__name__)


class FilterDetectionsController:
    """
    Controller class for filtering detections based on a specified threshold and source.

    Parameters
    ----------
    threshold : float
        The threshold value for filtering detections.
    filter_source : `morpheus.common.FilterSource`
        The source used for filtering.
    field_name : str
        The name of the field used for filtering.
    """

    def __init__(self, threshold: float, filter_source: FilterSource, field_name: str) -> None:
        self._threshold = threshold
        self._filter_source = filter_source
        self._field_name = field_name

    @property
    def threshold(self):
        """
        Get the threshold value.
        """
        return self._threshold

    @property
    def filter_source(self):
        """
        Get the filter source.
        """
        return self._filter_source

    @property
    def field_name(self):
        """
        Get the field name.
        """
        return self._field_name

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

        # If we didnt have any detections, return None
        if (true_pairs.shape[0] == 0):
            return None

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

    def update_filter_source(self, message_type: typing.Any):
        """
        This function updates filter source.

        Parameters
        ----------
        message_type : `typing.Any`
            Response message with probabilities calculated from inference results.
        """

        # Unfortunately we have to convert this to a list in case there are non-contiguous groups
        if self._filter_source == FilterSource.Auto:
            if (typing_utils.issubtype(message_type, MultiResponseMessage)):
                self._filter_source = FilterSource.TENSOR
            else:
                self._filter_source = FilterSource.DATAFRAME

            logger.debug(
                "filter_source was set to Auto, inferring a filter source of %s based on an input "
                "message type of %s",
                self._filter_source,
                message_type)
