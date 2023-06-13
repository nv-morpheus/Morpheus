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

import logging
import os
import queue
import time
import typing
from functools import partial
from io import StringIO

import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("to-rest-server", ignore_args=["df_serializer_fn"])
class RestServerSinkStage(SinglePortStage):
    """
    Write all messages to a REST endpoint.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    bind_address : str, default "127.0.0.1"
        The address to bind the REST server to.
    port : int, default 8080
        The port to bind the REST server to.
    endpoint : str, default "/"
        The endpoint to listen for requests on.
    method : str, default "GET"
        HTTP method to listen for.
    max_queue_size : int, default None
        Maximum number of requests to queue before rejecting requests. If `None` then `config.edge_buffer_size` will be
        used. Once the queue is full, the incoming edge buffer will begin to fill up.
    num_server_threads : int, default None
        Number of threads to use for the REST server. If `None` then `os.cpu_count()` will be used.
    max_rows_per_response : int, optional
        Maximum number of rows to include in a single response, by default 10000.
    overflow_pct: float, optional
        The stage stores incoming dataframes in a queue. If the received dataframes are smaller than
        `max_rows_per_response * overflow_pct`, then additional dataframes are popped off of the queue.
        Setting a higher number (0.9 or 1) can potentially improve performance by allowing as many dataframes to be
        concatinated as possible into a single response, but with the possibility of returning a response containing
        more than `max_rows_per_response` rows. Setting a lower number (0.5 or 0.75) decreases the chance, and a value
        of `0` prevents this possibility entirely.
    request_timeout_secs : int, default 30
        The maximum amount of time in seconds for any given request.
    lines : bool, default False
        If False, dataframes will be serialized to a JSON array of objects. If True, the dataframes will be
        serialized to a string JSON objects separated by end-of-line characters.
        Ignored if `df_serializer` is provided.
    df_serializer_fn : typing.Callable[[DataFrameType], str], optional
        Optional custom dataframe serializer function.
    """

    def __init__(
            self,
            config: Config,
            bind_address: str = "127.0.0.1",
            port: int = 8080,
            endpoint: str = "/message",
            method: str = "GET",
            max_queue_size: int = None,
            num_server_threads: int = None,
            max_rows_per_response: int = 10000,
            overflow_pct: float = 0.75,  # TODO: find a better name for this
            request_timeout_secs: int = 30,
            lines: bool = False,
            df_serializer_fn: typing.Callable[[DataFrameType], str] = None):
        super().__init__(config)
        self._max_rows_per_response = max_rows_per_response
        self._overflow_pct = overflow_pct
        self._request_timeout_secs = request_timeout_secs
        self._lines = lines

        if self._lines:
            self._content_type = MimeTypes.TEXT.value
        else:
            self._content_type = MimeTypes.JSON.value

        self._df_serializer_fn = df_serializer_fn or self._default_df_serializer

        from morpheus.common import RestServer

        # FiberQueue doesn't have a way to check the size, nor does it have a way to check if it's empty without
        # attempting to perform a read. We'll keep track of the size ourselves.
        self._queue = queue.Queue(maxsize=max_queue_size or config.edge_buffer_size)
        self._server = RestServer(parse_fn=self._request_handler,
                                  bind_address=bind_address,
                                  port=port,
                                  endpoint=endpoint,
                                  method=method,
                                  num_threads=num_server_threads or os.cpu_count(),
                                  request_timeout=request_timeout_secs)
        self._server.start()

    @property
    def name(self) -> str:
        return "to-rest-server"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        return False

    def _default_df_serializer(self, df: DataFrameType) -> str:
        """
        Default dataframe serializer, used when `df_serializer_fn` is not provided.
        """
        str_buf = StringIO()
        serializers.df_to_stream_json(df=df, stream=str_buf, lines=self._lines)
        str_buf.seek(0)
        return str_buf.read()

    def _request_callback(self, df: DataFrameType, num_tasks: int, has_error: bool, error_msg: str) -> None:
        try:
            if has_error:
                logger.error(error_msg)

                # If the client failed to read the response, then we need to put the dataframe back into the queue
                self._queue.put(df)

            # Even in the event of an error, we need to mark the tasks as done.
            for _ in range(num_tasks):
                self._queue.task_done()
        except Exception as e:
            logger.error("Unknown error in request callback: %s", e)

    def _request_handler(self, _: str) -> typing.Tuple[int, str]:
        # TODO: If this takes longer than `request_timeout_secs` then the request will be terminated, and the messages
        # will be lost
        num_rows = 0
        data_frames = []
        try:
            while (num_rows == 0 or num_rows < (self._max_rows_per_response * self._overflow_pct)):
                df = self._queue.get_nowait()
                num_rows += len(df)
                data_frames.append(df)
        except queue.Empty:
            pass
        except Exception as e:
            err_msg = "Unknown error processing request"
            logger.error(f"{err_msg}: %s", e)
            return (500, MimeTypes.TEXT.value, err_msg, None)

        if (len(data_frames) > 0):
            df = data_frames[0]
            if len(data_frames) > 1:
                cat_fn = pd.concat if isinstance(df, pd.DataFrame) else cudf.concat
                df = cat_fn(data_frames)

            return (200,
                    self._content_type,
                    self._df_serializer_fn(df),
                    partial(self._request_callback, df, len(data_frames)))
        else:
            return (204, MimeTypes.TEXT.value, "", None)

    def _partition_df(self, df: DataFrameType) -> typing.Iterable[DataFrameType]:
        """
        Partition a dataframe into slices no larger than `self._max_rows_per_response`.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to partition
        """
        slice_start = 0
        while (slice_start < len(df)):
            slice_end = min(slice_start + self._max_rows_per_response, len(df))
            df_slice = df.iloc[slice_start:slice_end]

            yield df_slice

            slice_start = slice_end

    def _process_message(self, msg: MessageMeta) -> MessageMeta:
        # In order to conform to the `self._max_rows_per_response` argument we need to slice up the dataframe here
        # because our queue isn't a deque.
        for df_slice in self._partition_df(msg.df):
            # We want to block, such that if the queue is full, we want our edgebuffer to start filling up.
            self._queue.put(df_slice, block=True)

        return msg

    def _block_until_empty(self):
        logger.debug("Waiting for queue to empty")
        self._queue.join()
        time.sleep(1)  # TODO: race condition, need some sort of on req callback, and only call task_done() there
        logger.debug("stopping server")
        self._server.stop()
        logger.debug("stopped")

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name,
                                 ops.map(self._process_message),
                                 ops.on_completed(self._block_until_empty))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
