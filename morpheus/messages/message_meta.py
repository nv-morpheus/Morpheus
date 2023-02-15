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
import warnings

import pandas as pd

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

    def __enter__(self):
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
class MessageMeta(MessageBase, cpp_class=_messages.MessageMeta):
    """
    This is a container class to hold batch deserialized messages metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Input rows in dataframe.

    """
    _df: pd.DataFrame = dataclasses.field(init=False)
    _mutex: threading.RLock = dataclasses.field(init=False, repr=False)

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

    def mutable_dataframe(self):
        return MutableTableCtxMgr(self)

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

    def has_unique_index(self) -> bool:
        """
        Returns True if the index of the underlying DataFrame is unique
        Returns
        -------
        bool
        """
        return self._df.index.is_unique

    def replace_non_unique_index(self):
        """
        Replaces the index in the underlying dataframe if the existing one is not unique.
        """

        if (not self.has_unique_index()):
            # Reset the index preserving the original index in a new column
            with self.mutable_dataframe() as df:
                # We could have had a race condition between calling has_unique_index() and acquiring the mutex.
                # Perform a second check here while we hold the lock.
                if (not df.index.is_unique):
                    logger.warning("Non unique index found in dataframe, generating new index.")
                    df.index.name = "_index_" + (df.index.name or "")
                    df.reset_index(inplace=True)


@dataclasses.dataclass(init=False)
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
    user_id: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, user_id: str) -> None:
        super().__init__(df)
        self.user_id = user_id


@dataclasses.dataclass(init=False)
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
    source: str = dataclasses.field(init=False)

    def __init__(self, df: pd.DataFrame, source: str) -> None:
        super().__init__(df)
        self.source = source
