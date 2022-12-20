# Copyright (c) 2022, NVIDIA CORPORATION.
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
import os
import pickle
import typing
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import DFPMessageMeta
from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.logging_timer import log_time

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


@dataclasses.dataclass
class CachedUserWindow:
    user_id: str
    cache_location: str
    timestamp_column: str = "timestamp"
    total_count: int = 0
    count: int = 0
    min_epoch: datetime = datetime(1970, 1, 1, tzinfo=timezone(timedelta(hours=0)))
    max_epoch: datetime = datetime(1970, 1, 1, tzinfo=timezone(timedelta(hours=0)))
    batch_count: int = 0
    pending_batch_count: int = 0
    last_train_count: int = 0
    last_train_epoch: datetime = None
    last_train_batch: int = 0

    _trained_rows: pd.Series = dataclasses.field(init=False, repr=False, default_factory=pd.DataFrame)
    _df: pd.DataFrame = dataclasses.field(init=False, repr=False, default_factory=pd.DataFrame)

    def append_dataframe(self, incoming_df: pd.DataFrame) -> bool:

        # # Get the row hashes
        # row_hashes = pd.util.hash_pandas_object(incoming_df)

        # Filter the incoming df by epochs later than the current max_epoch
        filtered_df = incoming_df[incoming_df["timestamp"] > self.max_epoch]

        if (len(filtered_df) == 0):
            # We have nothing new to add. Double check that we fit within the window
            before_history = incoming_df[incoming_df["timestamp"] < self.min_epoch]

            return len(before_history) == 0

        # Increment the batch count
        self.batch_count += 1
        self.pending_batch_count += 1

        # Set the filtered index
        filtered_df.index = range(self.total_count, self.total_count + len(filtered_df))

        # Save the row hash to make it easier to find later. Do this before the batch so it doesn't participate
        filtered_df["_row_hash"] = pd.util.hash_pandas_object(filtered_df, index=False)

        # Use batch id to distinguish groups in the same dataframe
        filtered_df["_batch_id"] = self.batch_count

        # Append just the new rows
        self._df = pd.concat([self._df, filtered_df])

        self.total_count += len(filtered_df)
        self.count = len(self._df)

        if (len(self._df) > 0):
            self.min_epoch = self._df[self.timestamp_column].min()
            self.max_epoch = self._df[self.timestamp_column].max()

        return True

    def get_train_df(self, max_history) -> pd.DataFrame:

        new_df = self.trim_dataframe(self._df,
                                     max_history=max_history,
                                     last_batch=self.batch_count - self.pending_batch_count,
                                     timestamp_column=self.timestamp_column)

        self.last_train_count = self.total_count
        self.last_train_epoch = datetime.now()
        self.last_train_batch = self.batch_count
        self.pending_batch_count = 0

        self._df = new_df

        if (len(self._df) > 0):
            self.min_epoch = self._df[self.timestamp_column].min()
            self.max_epoch = self._df[self.timestamp_column].max()

        return new_df

    def save(self):

        # Make sure the directories exist
        os.makedirs(os.path.dirname(self.cache_location), exist_ok=True)

        with open(self.cache_location, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def trim_dataframe(df: pd.DataFrame,
                       max_history: typing.Union[int, str],
                       last_batch: int,
                       timestamp_column: str = "timestamp") -> pd.DataFrame:
        if (max_history is None):
            return df

        # Want to ensure we always see data once. So any new data is preserved
        new_batches = df[df["_batch_id"] > last_batch]

        # See if max history is an int
        if (isinstance(max_history, int)):
            return df.tail(max(max_history, len(new_batches)))

        # If its a string, then its a duration
        if (isinstance(max_history, str)):
            # Get the latest timestamp
            latest = df[timestamp_column].max()

            time_delta = pd.Timedelta(max_history)

            # Calc the earliest
            earliest = min(latest - time_delta, new_batches[timestamp_column].min())

            return df[df[timestamp_column] >= earliest]

        raise RuntimeError("Unsupported max_history")

    @staticmethod
    def load(cache_location: str) -> "CachedUserWindow":

        with open(cache_location, "rb") as f:
            return pickle.load(f)


class DFPRollingWindowStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 min_history: int,
                 min_increment: int,
                 max_history: typing.Union[int, str],
                 cache_dir: str = "./.cache/dfp"):
        super().__init__(c)

        self._min_history = min_history
        self._min_increment = min_increment
        self._max_history = max_history
        self._cache_dir = os.path.join(cache_dir, "rolling-user-data")

        # Map of user ids to total number of messages. Keeps indexes monotonic and increasing per user
        self._user_cache_map: typing.Dict[str, CachedUserWindow] = {}

    @property
    def name(self) -> str:
        return "dfp-rolling-window"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (DFPMessageMeta, )

    def _trim_dataframe(self, df: pd.DataFrame):

        if (self._max_history is None):
            return df

        # See if max history is an int
        if (isinstance(self._max_history, int)):
            return df.tail(self._max_history)

        # If its a string, then its a duration
        if (isinstance(self._max_history, str)):
            # Get the latest timestamp
            latest = df[self._config.ae.timestamp_column_name].max()

            time_delta = pd.Timedelta(self._max_history)

            # Calc the earliest
            earliest = latest - time_delta

            return df[df['timestamp'] >= earliest]

        raise RuntimeError("Unsupported max_history")

    @contextmanager
    def _get_user_cache(self, user_id: str):

        # Determine cache location
        cache_location = os.path.join(self._cache_dir, f"{user_id}.pkl")

        user_cache = None

        user_cache = self._user_cache_map.get(user_id, None)

        if (user_cache is None):
            user_cache = CachedUserWindow(user_id=user_id,
                                          cache_location=cache_location,
                                          timestamp_column=self._config.ae.timestamp_column_name)

            self._user_cache_map[user_id] = user_cache

        yield user_cache

        # # When it returns, make sure to save
        # user_cache.save()

    def _build_window(self, message: DFPMessageMeta) -> MultiDFPMessage:

        user_id = message.user_id

        with self._get_user_cache(user_id) as user_cache:

            incoming_df = message.get_df()
            # existing_df = user_cache.df

            if (not user_cache.append_dataframe(incoming_df=incoming_df)):
                # Then our incoming dataframe wasnt even covered by the window. Generate warning
                logger.warn(("Incoming data preceeded existing history. "
                             "Consider deleting the rolling window cache and restarting."))
                return None

            # Exit early if we dont have enough data
            if (user_cache.count < self._min_history):
                return None

            # We have enough data, but has enough time since the last training taken place?
            if (user_cache.total_count - user_cache.last_train_count < self._min_increment):
                return None

            # Save the last train statistics
            train_df = user_cache.get_train_df(max_history=self._max_history)

            # Hash the incoming data rows to find a match
            incoming_hash = pd.util.hash_pandas_object(incoming_df.iloc[[0, -1]], index=False)

            # Find the index of the first and last row
            match = train_df[train_df["_row_hash"] == incoming_hash.iloc[0]]

            if (len(match) == 0):
                raise RuntimeError("Invalid rolling window")

            first_row_idx = match.index[0].item()
            last_row_idx = train_df[train_df["_row_hash"] == incoming_hash.iloc[-1]].index[-1].item()

            found_count = (last_row_idx - first_row_idx) + 1

            if (found_count != len(incoming_df)):
                raise RuntimeError(("Overlapping rolling history detected. "
                                    "Rolling history can only be used with non-overlapping batches"))

            train_offset = train_df.index.get_loc(first_row_idx)

            # Otherwise return a new message
            return MultiDFPMessage(meta=DFPMessageMeta(df=train_df, user_id=user_id),
                                   mess_offset=train_offset,
                                   mess_count=found_count)

    def on_data(self, message: DFPMessageMeta):

        with log_time(logger.debug) as log_info:

            result = self._build_window(message)

            if (result is not None):

                log_info.set_log(
                    ("Rolling window complete for %s in {duration:0.2f} ms. "
                     "Input: %s rows from %s to %s. Output: %s rows from %s to %s"),
                    message.user_id,
                    len(message.df),
                    message.df[self._config.ae.timestamp_column_name].min(),
                    message.df[self._config.ae.timestamp_column_name].max(),
                    result.mess_count,
                    result.get_meta(self._config.ae.timestamp_column_name).min(),
                    result.get_meta(self._config.ae.timestamp_column_name).max(),
                )
            else:
                # Dont print anything
                log_info.disable()

            return result

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiDFPMessage
