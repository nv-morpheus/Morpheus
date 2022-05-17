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

import dataclasses

import cupy as cp

from morpheus.messages import DataClassProp
from morpheus.messages import InferenceMemory
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ResponseMemory
from morpheus.messages import get_input
from morpheus.messages import get_output
from morpheus.messages import set_input
from morpheus.messages import set_output


@dataclasses.dataclass
class ResponseMemoryLogParsing(ResponseMemory, cpp_class=None):

    confidences: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)
    labels: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)

    def __post_init__(self, confidences, labels):
        self.confidences = confidences
        self.labels = labels


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
class PostprocMemoryLogParsing(InferenceMemory):
    """
    This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.

    Parameters
    ----------
    confidences: cp.ndarray
        confidences calculated from softmax
    labels: cp.ndarray
        index of highest confidence
    input_ids : cp.ndarray
        The token-ids for each string padded with 0s to max_length.
    seq_ids : cp.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e. If some messages get broken into multiple inference requests)

    """

    confidences: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    labels: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    input_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __post_init__(self, confidences, labels, input_ids, seq_ids):
        self.confidences = confidences
        self.labels = labels
        self.input_ids = input_ids
        self.seq_ids = seq_ids


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
