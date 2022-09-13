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
import typing

from dfencoder.autoencoder import AutoEncoder

from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_inference_message import MultiInferenceMessage


@dataclasses.dataclass
class MultiInferenceAEMessage(MultiInferenceMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for AE workloads. Helps ensure the
    proper inputs are set and eases debugging. Associates a user ID with a message.
    """

    model: AutoEncoder
    # train_loss_scores: cp.ndarray
    train_scores_mean: float
    train_scores_std: float

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

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MultiInferenceAEMessage`
            A new `MultiInferenceAEMessage` with sliced offset and count.

        """
        mess_start = self.mess_offset + self.seq_ids[start, 0].item()
        mess_stop = self.mess_offset + self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceAEMessage(meta=self.meta,
                                       mess_offset=mess_start,
                                       mess_count=mess_stop - mess_start,
                                       memory=self.memory,
                                       offset=start,
                                       count=stop - start,
                                       model=self.model,
                                       train_scores_mean=self.train_scores_mean,
                                       train_scores_std=self.train_scores_std)
