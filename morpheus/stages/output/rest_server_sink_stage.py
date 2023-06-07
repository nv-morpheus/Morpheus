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
import typing
from io import StringIO

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils import http_utils
from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


@register_stage("rest-server-sink", ignore_args=["df_serializer"])
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
    request_timeout_secs : int, default 30
        The maximum amount of time in seconds for any given request.
    lines : bool, default False
        If False, dataframes will be serialized to a JSON array of objects. If True, the dataframes will be
        serialized to a string JSON objects separated by end-of-line characters.
        Ignored if `df_serializer` is provided.
    df_serializer_fn : typing.Callable[[DataFrameType], str], optional
        Optional custom dataframe serializer function.
    """

    def __init__(self,
                 config: Config,
                 bind_address: str = "127.0.0.1",
                 port: int = 8080,
                 endpoint: str = "/message",
                 method: str = "GET",
                 max_queue_size: int = None,
                 num_server_threads: int = None,
                 max_rows_per_response: int = 10000,
                 request_timeout_secs: int = 30,
                 lines: bool = False,
                 df_serializer_fn: typing.Callable[[DataFrameType], str] = None):
        super().__init__(config)
        self._max_rows_per_response = max_rows_per_response
        self._request_timeout_secs = request_timeout_secs
        self._lines = lines
        self._df_serializer_fn = df_serializer_fn or self._default_df_serializer

        from morpheus.common import FiberQueue
        from morpheus.common import RestServer

        self._queue = FiberQueue(max_queue_size or config.edge_buffer_size)
        self._server = RestServer(parse_fn=self._parse_payload,
                                  bind_address=bind_address,
                                  port=port,
                                  endpoint=endpoint,
                                  method=method,
                                  num_threads=num_server_threads or os.cpu_count(),
                                  request_timeout=request_timeout_secs)

    @property
    def name(self) -> str:
        return "rest-server-sink"

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
            slice_end = min(slice_start + self._max_rows_per_payload, len(df))
            df_slice = df.iloc[slice_start:slice_end]

            yield df_slice

            slice_start = slice_end

    def _process_message(self, msg: MessageMeta) -> MessageMeta:

        # In order to conform to the `self._max_rows_per_response` argument we need to slice up the dataframe here
        # because our queue isn't a deque.
        for df_slice in self._partition_df(msg.df):
            self._queue.put(df_slice)

        return msg

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
