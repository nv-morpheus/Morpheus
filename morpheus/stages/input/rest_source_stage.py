# Copyright (c) 2023, NVIDIA CORPORATION.
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

import mrc

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("POST", "PUT")


@register_stage("from-rest")
class RestSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that starts an HTTP server and listens for incoming REST requests on a specified endpoint.

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
    method : str, default "POST"
        HTTP method to listen for. Valid values are "POST" and "PUT".
    sleep_time : float, default 0.1
        Amount of time in seconds to sleep if the request queue is empty.
    queue_timeout : int, default 5
        Maximum amount of time in seconds to wait for a request to be added to the queue before rejecting requests.
    max_queue_size : int, default None
        Maximum number of requests to queue before rejecting requests. If `None` then `config.edge_buffer_size` will be
        used.
    num_server_threads : int, default None
        Number of threads to use for the REST server. If `None` then `os.cpu_count()` will be used.
    max_payload_size : int, default 10
        The maximum size in megabytes of the payload that the server will accept in a single request.
    request_timeout_secs : int, default 30
        The maximum amount of time in seconds for any given request.
    lines : bool, default False
        If False, the REST server will expect each request to be a JSON array of objects. If True, the REST server will
        expect each request to be a JSON object per line.
    """

    def __init__(self,
                 config: Config,
                 bind_address: str = "127.0.0.1",
                 port: int = 8080,
                 endpoint: str = "/message",
                 method: str = "POST",
                 sleep_time: float = 0.1,
                 queue_timeout: int = 5,
                 max_queue_size: int = None,
                 num_server_threads: int = None,
                 max_payload_size: int = 10,
                 request_timeout_secs: int = 30,
                 lines: bool = False):
        super().__init__(config)
        self._bind_address = bind_address
        self._port = port
        self._endpoint = endpoint
        self._method = method
        self._sleep_time = sleep_time
        self._queue_timeout = queue_timeout
        self._max_queue_size = max_queue_size or config.edge_buffer_size
        self._num_server_threads = num_server_threads or os.cpu_count()
        self._max_payload_size_bytes = max_payload_size * 1024 * 1024
        self._request_timeout_secs = request_timeout_secs
        self._lines = lines

        # This is only used when C++ mode is disabled
        self._queue = None

        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")

    @property
    def name(self) -> str:
        return "from-rest"

    def supports_cpp_node(self) -> bool:
        return True

    def _parse_payload(self, payload: str) -> typing.Tuple[int, str]:
        try:
            # engine='cudf' is needed when lines=False to avoid using pandas
            df = cudf.read_json(payload, lines=self._lines, engine='cudf')
        except Exception as e:
            err_msg = "Error occurred converting REST payload to Dataframe"
            logger.error(f"{err_msg}: {e}")
            return (400, MimeTypes.TEXT.value, err_msg, None)

        try:
            self._queue.put(df, block=True, timeout=self._queue_timeout)
            return (201, MimeTypes.TEXT.value, "", None)
        except (queue.Full, Closed) as e:
            err_msg = "REST payload queue is "
            if isinstance(e, queue.Full):
                err_msg += "full"
            else:
                err_msg += "closed"
            logger.error(err_msg)
            return (503, MimeTypes.TEXT.value, err_msg, None)
        except Exception as e:
            err_msg = "Error occurred while pushing payload to queue"
            logger.error(f"{err_msg}: {e}")
            return (500, MimeTypes.TEXT.value, err_msg, None)

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        from morpheus.common import FiberQueue
        from morpheus.common import RestServer

        self._queue = FiberQueue(self._max_queue_size)
        rest_server = RestServer(parse_fn=self._parse_payload,
                                 bind_address=self._bind_address,
                                 port=self._port,
                                 endpoint=self._endpoint,
                                 method=self._method,
                                 num_threads=self._num_server_threads,
                                 max_payload_size=self._max_payload_size_bytes,
                                 request_timeout=self._request_timeout_secs)
        rest_server.start()

        processing = True
        while (processing):
            # Read as many messages as we can from the queue if it's empty check to see if we should be shutting down
            # It is important that any messages we received that are in the queue are processed before we shutdown since
            # we already returned an OK response to the client.
            df = None
            try:
                df = self._queue.get()
            except queue.Empty:
                if (not rest_server.is_running()):
                    processing = False
                else:
                    logger.debug("Queue empty, sleeping ...")
                    time.sleep(self._sleep_time)
            except Closed:
                logger.error("Queue closed unexpectedly, shutting down")
                processing = False

            if df is not None:
                yield MessageMeta(df)

        logger.debug("Stopping REST server ...")
        rest_server.stop()
        logger.debug("REST server stopped")
        self._queue.close()

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        if self._build_cpp_node():
            import morpheus._lib.stages as _stages
            node = _stages.RestSourceStage(builder,
                                           self.unique_name,
                                           bind_address=self._bind_address,
                                           port=self._port,
                                           endpoint=self._endpoint,
                                           method=self._method,
                                           sleep_time=self._sleep_time,
                                           queue_timeout=self._queue_timeout,
                                           max_queue_size=self._max_queue_size,
                                           num_server_threads=self._num_server_threads,
                                           max_payload_size=self._max_payload_size_bytes,
                                           request_timeout=self._request_timeout_secs,
                                           lines=self._lines)
        else:
            node = builder.make_source(self.unique_name, self._generate_frames())

        return node, MessageMeta
