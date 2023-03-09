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
import logging
import typing

from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_message import MultiMessage
from morpheus.models.dfencoder import AutoEncoder

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiAEMessage(MultiMessage):
    """
    Subclass of `MultiMessage` specific to the AutoEncoder pipeline, which contains the model.
    """

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

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message instance for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The final output is exclusive of the `stop_row`, i.e. `[start_row, stop_row)`. For example to copy rows
            1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        Returns
        -------
        `MultiAEMessage`
        """

        sliced_rows = self.copy_meta_ranges(ranges)
        return MultiAEMessage(meta=UserMessageMeta(sliced_rows, user_id=self.meta.user_id),
                              mess_offset=0,
                              mess_count=len(sliced_rows),
                              model=self.model,
                              train_scores_mean=self.train_scores_mean,
                              train_scores_std=self.train_scores_std)
