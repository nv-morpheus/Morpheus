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
import logging

from dfencoder import AutoEncoder

from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_message import MultiMessage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiAEMessage(MultiMessage):

    model: AutoEncoder
    # train_loss_scores: cp.ndarray
    train_scores_mean: float = 0.0
    train_scores_std: float = 1.0

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
        morpheus.pipeline.preprocess.autoencoder.MultiAEMessage
            A new `MultiAEMessage` with sliced offset and count.

        """
        return MultiAEMessage(meta=self.meta,
                              mess_offset=start,
                              mess_count=stop - start,
                              model=self.model,
                              train_scores_mean=self.train_scores_mean,
                              train_scores_std=self.train_scores_std)

    def copy_ranges(self, ranges, num_selected_rows=None):
        sliced_rows = self.copy_meta_ranges(ranges)

        if num_selected_rows is None:
            num_selected_rows = len(sliced_rows)

        return MultiAEMessage(meta=UserMessageMeta(sliced_rows, user_id=self.meta.user_id),
                              mess_offset=0,
                              mess_count=num_selected_rows,
                              model=self.model,
                              train_scores_mean=self.train_scores_mean,
                              train_scores_std=self.train_scores_std)
