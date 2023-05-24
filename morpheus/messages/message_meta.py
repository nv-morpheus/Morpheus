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
import threading
import typing
import warnings

import pandas as pd

import cudf

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageBase

logger = logging.getLogger(__name__)


class MutableTableCtxMgr:
    """
    Context manager for editing the DataFrame held by a MessageMeta, ensures an editing lock aqcuired and released.
    Not intended to be used directly but is instead invoked via MessageMeta's `mutable_dataframe`.

    Examples
    --------
    >>> with meta.mutable_dataframe() as df:
    >>>     df['col'] = 5
    """

    ussage_error = ("Error attempting to use mutable_dataframe outside of context manager. Intended usage :\n"
                    "with message_meta.mutable_dataframe() as df:\n"
                    "    df['col'] = 5")

    def __init__(self, meta) -> None:
        self.__dict__['__meta'] = meta

    def __enter__(self) -> pd.DataFrame:
        meta = self.__dict__['__meta']
        meta._mutex.acquire()
        return meta._df

    def __exit__(self, exc_type, exc_value, traceback):
        self.__dict__['__meta']._mutex.release()

    def __getattr__(self, name):
        raise AttributeError(self.ussage_error)

    def __getitem__(self, key):
        raise AttributeError(self.ussage_error)

    def __setattr__(self, name, value):
        raise AttributeError(self.ussage_error)

    def __setitem__(self, key, value):
        raise AttributeError(self.ussage_error)


@dataclasses.dataclass(init=False)
class MessageMeta(_messages.MessageMeta):
    """
    This is a container class to hold batch deserialized messages metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def get_meta_range(self,
                       mess_offset: int,
                       message_count: int,
                       columns: typing.Union[None, str, typing.List[str]] = None):
        """
        Return column values from `morpheus.pipeline.messages.MessageMeta.df` from the specified start offset
        until the message count.

        Parameters
        ----------
        mess_offset : int
            Offset into the metadata batch.
        mess_count : int
            Messages count.
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Returns all columns if `None` is specified. When a string is passed, a `Series` is
            returned. Otherwise a `Dataframe` is returned.

        Returns
        -------
        Series or Dataframe
            Column values from the dataframe.

        """

        idx = self.df.index[mess_offset:mess_offset + message_count]

        if (isinstance(idx, cudf.RangeIndex)):
            idx = slice(idx.start, idx.stop - 1, idx.step)

        if (columns is None):
            return self.df.loc[idx, :]
        else:
            # If its a str or list, this is the same
            return self.df.loc[idx, columns]


@dataclasses.dataclass(init=False)
class UserMessageMeta(MessageMeta):
    """
    This class extends MessageMeta to also hold userid corresponding to batched metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.
    user_id : str
        User id.

    """
    user_id: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, user_id: str) -> None:
        super().__init__(df)
        self.user_id = user_id


@dataclasses.dataclass(init=False)
class AppShieldMessageMeta(MessageMeta):
    """
    This class extends MessageMeta to also hold source corresponding to batched metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows in dataframe.
    source : str
        Determines which source generated the snapshot messages.
    """
    source: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, source: str) -> None:
        super().__init__(df)
        self.source = source
