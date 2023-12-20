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

import dataclasses

import cupy as cp

from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseMessage


@dataclasses.dataclass
class MultiResponseLogParsingMessage(MultiResponseMessage, cpp_class=None):
    """
    A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
    array. Helps ensure the proper outputs are set and eases debugging.
    """

    @property
    def confidences(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self.get_output("confidences")

    @property
    def labels(self):
        """
        Returns sequence ids, which are used to keep track of which inference requests belong to each message.

        Returns
        -------
        cupy.ndarray
            Ids used to index from an inference input to a message. Necessary since there can be more
            inference inputs than messages (i.e. If some messages get broken into multiple inference requests)

        """

        return self.get_output("labels")

    @property
    def input_ids(self):
        """
        input_ids

        Returns
        -------
        cp.ndarray
            input_ids

        """

        return self.get_output("input_ids")

    @property
    def seq_ids(self):
        """
        seq_ids

        Returns
        -------
        cp.ndarray
            seq_ids

        """

        return self.get_output("seq_ids")


@dataclasses.dataclass
class MultiPostprocLogParsingMessage(MultiInferenceMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    @property
    def confidences(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self.get_input("confidences")

    @property
    def labels(self):
        """
        Returns sequence ids, which are used to keep track of which inference requests belong to each message.

        Returns
        -------
        cupy.ndarray
            Ids used to index from an inference input to a message. Necessary since there can be more
            inference inputs than messages (i.e. If some messages get broken into multiple inference requests)

        """

        return self.get_input("labels")

    @property
    def input_ids(self):
        """
        input_ids

        Returns
        -------
        cp.ndarray
            input_ids

        """

        return self.get_input("input_ids")

    @property
    def seq_ids(self):
        """
        seq_ids

        Returns
        -------
        cp.ndarray
            seq_ids

        """

        return self.get_input("seq_ids")
