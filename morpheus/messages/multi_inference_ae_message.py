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
import typing

from dfencoder.autoencoder import AutoEncoder

from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_inference_message import MultiInferenceMessage


@dataclasses.dataclass
class MultiInferenceAEMessage(MultiInferenceMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for AE workloads. Helps ensure the
    proper inputs are set and eases debugging. Associates a user ID with a message.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["seq_ids"]

    model: AutoEncoder
    # train_loss_scores: cp.ndarray
    train_scores_mean: float
    train_scores_std: float

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1,
                 model: AutoEncoder = None,
                 train_scores_mean: float = float("NaN"),
                 train_scores_std: float = float("NaN")):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count)

        self.model = model
        self.train_scores_mean = train_scores_mean
        self.train_scores_std = train_scores_std

    @property
    def user_id(self):
        """
        Returns the user ID associated with this message.

        """

        return typing.cast(UserMessageMeta, self.meta).user_id

    @property
    def input(self):
        """
        Returns autoecoder input tensor.

        Returns
        -------
        cupy.ndarray
            The autoencoder input tensor.

        """

        return self.get_input("input")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            seq_ids

        """

        return self.get_input("seq_ids")
