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
import threading
import warnings
from contextlib import contextmanager

import pandas as pd

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageBase


@dataclasses.dataclass(init=False)
class MessageMeta(MessageBase, cpp_class=_messages.MessageMeta):
    """
    This is a container class to hold batch deserialized messages metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.

    """
    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._mutex = threading.RLock()
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        msg = ("Warning the df property returns a copy, please use the copy_dataframe method or the mutable_dataframe "
               "context manager to modify the DataFrame in-place instead.")

        warnings.warn(msg, DeprecationWarning)
        return self.copy_dataframe()

    def copy_dataframe(self):
        return self._df.copy(deep=True)

    @contextmanager
    def mutable_dataframe(self):
        self._mutex.acquire()
        try:
            yield self._df
        finally:
            self._mutex.release()

    @property
    def count(self) -> int:
        """
        Returns the number of messages in the batch.

        Returns
        -------
        int
            number of messages in the MessageMeta.df.

        """

        return len(self._df)


@dataclasses.dataclass
class UserMessageMeta(MessageMeta, cpp_class=None):
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


@dataclasses.dataclass
class AppShieldMessageMeta(MessageMeta, cpp_class=None):
    """
    This class extends MessageMeta to also hold source corresponding to batched metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows in dataframe.
    source : str
        Determines which source generated the snapshot messages.
    """
    source: str
