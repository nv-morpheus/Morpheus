# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
"""Source stage that starts an HTTP server and listens for incoming requests on a specified endpoint."""

import logging
import os
import queue
import time
import typing
from http import HTTPStatus

import mrc

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.utils.http_utils import HTTPMethod
from morpheus.utils.http_utils import HttpParseResponse
from morpheus.utils.http_utils import MimeTypes
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = (HTTPMethod.POST, HTTPMethod.PUT)
HEALTH_SUPPORTED_METHODS = (HTTPMethod.GET, HTTPMethod.POST)


@register_stage("from-http")
class HttpServerSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that starts an HTTP server and listens for incoming requests on a specified endpoint.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    bind_address : str, default "127.0.0.1"
        The address to bind the HTTP server to.
    port : int, default 8080
        The port to bind the HTTP server to.
    endpoint : str, default "/"
        The endpoint to listen for requests on.
    method : `morpheus.utils.http_utils.HTTPMethod`, optional, case_sensitive = False
        HTTP method to listen for. Valid values are "POST" and "PUT".
    accept_status: `http.HTTPStatus`, default 201, optional
        The HTTP status code to return when a request is accepted. Valid values must be in the 2xx range.
    sleep_time : float, default 0.1
        Amount of time in seconds to sleep if the request queue is empty.
    queue_timeout : int, default 5
        Maximum amount of time in seconds to wait for a request to be added to the queue before rejecting requests.
    max_queue_size : int, default None
        Maximum number of requests to queue before rejecting requests. If `None` then `config.edge_buffer_size` will be
        used.
    num_server_threads : int, default None
        Number of threads to use for the HTTP server. If `None` then `os.cpu_count()` will be used.
    max_payload_size : int, default 10
        The maximum size in megabytes of the payload that the server will accept in a single request.
    request_timeout_secs : int, default 30
        The maximum amount of time in seconds for any given request.
    lines : bool, default False
        If False, the HTTP server will expect each request to be a JSON array of objects. If True, the HTTP server will
        expect each request to be a JSON object per line.
    stop_after : int, default 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    payload_to_df_fn : callable, default None
        A callable that takes the HTTP payload string as the first argument and the `lines` parameter is passed in as
        the second argument and returns a cudf.DataFrame. When supplied, the C++ implementation of this stage is
        disabled, and the Python impl is used.
    """

    def __init__(self,
                 config: Config,
                 bind_address: str = "127.0.0.1",
                 port: int = 8080,
                 endpoint: str = "/message",
                 live_endpoint: str = "/live",
                 ready_endpoint: str = "/ready",
                 method: HTTPMethod = HTTPMethod.POST,
                 live_method: HTTPMethod = HTTPMethod.GET,
                 ready_method: HTTPMethod = HTTPMethod.GET,
                 accept_status: HTTPStatus = HTTPStatus.CREATED,
                 sleep_time: float = 0.1,
                 queue_timeout: int = 5,
                 max_queue_size: int = None,
                 num_server_threads: int = None,
                 max_payload_size: int = 10,
                 request_timeout_secs: int = 30,
                 lines: bool = False,
                 stop_after: int = 0,
                 payload_to_df_fn: typing.Callable[[str, bool], cudf.DataFrame] = None):
        super().__init__(config)
        self._bind_address = bind_address
        self._port = port
        self._endpoint = endpoint
        self._live_endpoint = live_endpoint
        self._ready_endpoint = ready_endpoint
        self._method = method
        self._live_method = live_method
        self._ready_method = ready_method
        self._accept_status = accept_status
        self._sleep_time = sleep_time
        self._queue_timeout = queue_timeout
        self._max_queue_size = max_queue_size or config.edge_buffer_size
        self._num_server_threads = num_server_threads or os.cpu_count()
        self._max_payload_size_bytes = max_payload_size * 1024 * 1024
        self._request_timeout_secs = request_timeout_secs
        self._lines = lines
        self._stop_after = stop_after
        self._payload_to_df_fn = payload_to_df_fn
        self._http_server = None

        # These are only used when C++ mode is disabled
        self._queue = None
        self._queue_size = 0
        self._processing = False
        self._records_emitted = 0

        for url, http_method in [(endpoint, method), (live_endpoint, live_method), (ready_endpoint, ready_method)]:
            check_supported_method(url, http_method, SUPPORTED_METHODS if url == endpoint else HEALTH_SUPPORTED_METHODS)

        if accept_status.value < 200 or accept_status.value > 299:
            raise ValueError(f"Invalid accept_status: {accept_status}")

    @property
    def name(self) -> str:
        """Unique name of the stage."""
        return "from-http"

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports C++ nodes."""
        return True

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(MessageMeta)

    def _parse_payload(self, payload: str) -> HttpParseResponse:
        try:
            if self._payload_to_df_fn is not None:
                df = self._payload_to_df_fn(payload, self._lines)
            else:
                # engine='cudf' is needed when lines=False to avoid using pandas
                df = cudf.read_json(payload, lines=self._lines, engine='cudf')

        except Exception as e:
            err_msg = "Error occurred converting HTTP payload to Dataframe"
            logger.error("%s: %s", err_msg, e)
            return HttpParseResponse(status_code=HTTPStatus.BAD_REQUEST.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

        try:
            self._queue.put(df, block=True, timeout=self._queue_timeout)
            self._queue_size += 1
            return HttpParseResponse(status_code=self._accept_status.value, content_type=MimeTypes.TEXT.value, body="")

        except (queue.Full, Closed) as e:
            err_msg = "HTTP payload queue is "
            if isinstance(e, queue.Full):
                err_msg += "full"
            else:
                err_msg += "closed"
            logger.error(err_msg)
            return HttpParseResponse(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

        except Exception as e:
            err_msg = "Error occurred while pushing payload to queue"
            logger.error("%s: %s", err_msg, e)
            return HttpParseResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

    def _liveliness_check(self, _: str) -> HttpParseResponse:
        if not self._http_server.is_running():
            err_msg = "Source server is not running"
            logger.error(err_msg)
            return HttpParseResponse(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

        return HttpParseResponse(status_code=self._accept_status.value, content_type=MimeTypes.TEXT.value, body="")

    def _readiness_check(self, _: str) -> HttpParseResponse:
        if not self._http_server.is_running():
            err_msg = "Source server is not running"
            logger.error(err_msg)
            return HttpParseResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                                     content_type=MimeTypes.TEXT.value,
                                     body=err_msg)

        if self._queue_size < self._max_queue_size:
            return HttpParseResponse(status_code=self._accept_status.value, content_type=MimeTypes.TEXT.value, body="")

        err_msg = "HTTP payload queue is full or unavailable to accept new values"
        logger.error(err_msg)
        return HttpParseResponse(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                                 content_type=MimeTypes.TEXT.value,
                                 body=err_msg)

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        from morpheus.common import FiberQueue
        from morpheus.common import HttpEndpoint
        from morpheus.common import HttpServer

        msg = HttpEndpoint(self._parse_payload, self._endpoint, self._method.name)
        live = HttpEndpoint(self._liveliness_check, self._live_endpoint, self._live_method.name)
        ready = HttpEndpoint(self._readiness_check, self._ready_endpoint, self._ready_method.name)
        with (FiberQueue(self._max_queue_size) as self._queue,
              HttpServer(endpoints=[msg, live, ready],
                         bind_address=self._bind_address,
                         port=self._port,
                         num_threads=self._num_server_threads,
                         max_payload_size=self._max_payload_size_bytes,
                         request_timeout=self._request_timeout_secs) as http_server):

            self._processing = True
            self._http_server = http_server
            while self._processing:
                # Read as many messages as we can from the queue if it's empty check to see if we should be shutting
                # down. It is important that any messages we received that are in the queue are processed before we
                # shutdown since we already returned an OK response to the client.
                df = None
                try:
                    df = self._queue.get()
                    self._queue_size -= 1
                except queue.Empty:
                    if (not self._http_server.is_running()):
                        self._processing = False
                    else:
                        logger.debug("Queue empty, sleeping ...")
                        time.sleep(self._sleep_time)
                except Closed:
                    logger.error("Queue closed unexpectedly, shutting down")
                    self._processing = False

                if df is not None:
                    num_records = len(df)
                    yield MessageMeta(df)
                    self._records_emitted += num_records

                    if self._stop_after > 0 and self._records_emitted >= self._stop_after:
                        self._processing = False

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        if self._build_cpp_node() and self._payload_to_df_fn is None:
            import morpheus._lib.stages as _stages
            node = _stages.HttpServerSourceStage(builder,
                                                 self.unique_name,
                                                 bind_address=self._bind_address,
                                                 port=self._port,
                                                 endpoint=self._endpoint,
                                                 method=self._method.value,
                                                 accept_status=self._accept_status.value,
                                                 sleep_time=self._sleep_time,
                                                 queue_timeout=self._queue_timeout,
                                                 max_queue_size=self._max_queue_size,
                                                 num_server_threads=self._num_server_threads,
                                                 max_payload_size=self._max_payload_size_bytes,
                                                 request_timeout=self._request_timeout_secs,
                                                 lines=self._lines,
                                                 stop_after=self._stop_after)
        else:
            node = builder.make_source(self.unique_name, self._generate_frames())

        return node


def check_supported_method(endpoint: str, method: HTTPMethod, supported_methods: typing.Tuple[HTTPMethod, ...]) -> None:
    if method not in supported_methods:
        raise ValueError(
            f"Unsupported method {method} for endpoint {endpoint}. Supported methods are {supported_methods}")
