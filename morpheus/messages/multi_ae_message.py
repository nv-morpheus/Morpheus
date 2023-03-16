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

from dfencoder import AutoEncoder

from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_message import MultiMessage

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

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 model: AutoEncoder,
                 train_scores_mean: float,
                 train_scores_std: float):
        super().__init__(meta=meta, mess_offset=mess_offset, mess_count=mess_count)

        self.model = model
        self.train_scores_mean = train_scores_mean
        self.train_scores_std = train_scores_std
