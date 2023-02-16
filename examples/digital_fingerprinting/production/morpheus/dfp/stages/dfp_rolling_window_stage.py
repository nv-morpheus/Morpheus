# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import logging
import os
import typing
from contextlib import contextmanager

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import DFPMessageMeta
from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.cached_user_window import CachedUserWindow
from ..utils.logging_timer import log_time

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


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

            # Otherwise return a new message
            return MultiDFPMessage(meta=DFPMessageMeta(df=train_df, user_id=user_id),
                                   mess_offset=0,
                                   mess_count=len(train_df))

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
