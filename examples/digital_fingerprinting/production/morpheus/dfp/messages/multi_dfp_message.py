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
import typing

from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DFPMessageMeta(MessageMeta, cpp_class=None):
    """
    This class extends MessageMeta to also hold userid corresponding to batched metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.
    user_id : str
        User id.

    """
    user_id: str

    def get_df(self):
        return self.df

    def set_df(self, df):
        self.df = df


@dataclasses.dataclass
class MultiDFPMessage(MultiMessage):

    def __post_init__(self):

        assert isinstance(self.meta, DFPMessageMeta), "`meta` must be an instance of DFPMessageMeta"

    @property
    def user_id(self):
        return typing.cast(DFPMessageMeta, self.meta).user_id

    def get_meta_dataframe(self):
        return typing.cast(DFPMessageMeta, self.meta).get_df()

    def set_meta_dataframe(self, columns: typing.Union[None, str, typing.List[str]], value):

        df = typing.cast(DFPMessageMeta, self.meta).get_df()

        if (columns is None):
            # Set all columns
            df[list(value.columns)] = value
        else:
            # If its a single column or list of columns, this is the same
            df[columns] = value

        typing.cast(DFPMessageMeta, self.meta).set_df(df)

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
        return MultiDFPMessage(meta=self.meta, mess_offset=start, mess_count=stop - start)
