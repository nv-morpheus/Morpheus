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
import time
import typing

import mrc
import requests

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils import http_utils

logger = logging.getLogger(__name__)


@register_stage("from-rest-client", ignore_args=["query_params", "headers", "**request_kwargs"])
class RestClientSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that polls a remote HTTP server for incoming data.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    url : str
        Remote URL to poll for data, ex `https://catalog.ngc.nvidia.com/api/collections`. This should include protocol
        prefix (ex. "http://", "https://") and port if necessary. If the protocol is omitted, `http://` will be used.
    query_params : dict, callable, default None
        Query parameters to pass to the remote URL. Can either be a dictionary of key-value pairs or a callable that
        returns a dictionary of key-value pairs. If a callable is provided, it will be called with no arguments.
    headers: dict, default None
        Headers sent with the request.
    method : str, default "GET"
        HTTP method to use.
    sleep_time : float, default 0.1
        Amount of time in seconds to sleep between successive requests. Setting this to 0 disables this feature.
    error_sleep_time : float, default 0.1
        Amount of time in seconds to sleep after the client receives an error.
        The client will perform an exponential backoff starting at `error_sleep_time`.
        Setting this to 0 causes the client to retry the request as fast as possible.
        If the server sets a `Retry-After` header and `respect_retry_after_header` is `True`, then that value will take
        precedence over `error_sleep_time`.
    respect_retry_after_header: bool, default True
        If True, the client will respect the `Retry-After` header if it is set by the server. If False, the client will
        perform an exponential backoff starting at `error_sleep_time`.
    request_timeout_secs : int, optional
        Number of seconds to wait for the server to send data before giving up and raising an exception.
    max_errors : int, default 10
        Maximum number of consequtive errors to receive before raising an error.
    accept_status_codes : typing.List[int], optional,  multiple = True
        List of status codes to accept. If the response status code is not in this tuple, then the request will be
        considered an error
    max_retries : int, default 10
        Maximum number of times to retry the request fails, receives a redirect or returns a status in the
        `retry_status_codes` list. Setting this to 0 disables this feature, and setting this to a negative number will raise
        a `ValueError`.
    stop_after: int, default = 0
        Stops ingesting after emitting `stop_after` records (rows in the dataframe). Useful for testing. Disabled if `0`
    lines : bool, default False
        If False, the response payloads are expected to be a JSON array of objects. If True, the payloads are expected
        to contain a JSON objects separated by end-of-line characters.
    **request_kwargs : dict
        Additional arguments to pass to the `requests.request` function.
    """

    def __init__(self,
                 config: Config,
                 url: str,
                 query_params: typing.Union[dict, typing.Callable] = None,
                 headers: dict = None,
                 method: str = "GET",
                 sleep_time: float = 0.1,
                 error_sleep_time: float = 0.1,
                 respect_retry_after_header: bool = True,
                 request_timeout_secs: int = 30,
                 accept_status_codes: typing.List[int] = (200, ),
                 max_retries: int = 10,
                 stop_after: int = 0,
                 lines: bool = False,
                 **request_kwargs):
        super().__init__(config)
        self._url = http_utils.verify_url(url)

        if callable(query_params):
            self._query_params_fn = query_params
            self._query_params = None
        else:
            self._query_params_fn = None
            self._query_params = query_params

        self._headers = headers

        self._method = method

        if sleep_time >= 0:
            self._sleep_time = sleep_time
        else:
            raise ValueError("sleep_time must be >= 0")

        if error_sleep_time >= 0:
            self._error_sleep_time = error_sleep_time
        else:
            raise ValueError("error_sleep_time must be >= 0")

        self._respect_retry_after_header = respect_retry_after_header

        self._request_timeout_secs = request_timeout_secs
        if max_retries >= 0:
            self._max_retries = max_retries
        else:
            raise ValueError("max_retries must be >= 0")

        self._accept_status_codes = tuple(accept_status_codes)

        self._stop_after = stop_after
        self._lines = lines
        self._requst_kwargs = request_kwargs

    @property
    def name(self) -> str:
        return "from-rest-client"

    def supports_cpp_node(self) -> bool:
        return False

    def _parse_response(self, response: requests.Response) -> typing.Union[cudf.DataFrame, None]:
        """
        Returns a DataFrame parsed from the response payload. If the response payload is empty, then `None` is returned.
        """
        payload = response.content
        if len(payload) > 2:  # work-around for https://github.com/rapidsai/cudf/issues/5712
            return cudf.read_json(payload, lines=self._lines, engine='cudf')
        else:
            return None

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        # Running counter of the number of messages emitted by this source
        num_messages = 0

        # The http_session variable is an in/out argument for the requests_retry_wrapper.request function and will be
        # initialized on the first call
        http_session = None

        request_args = {
            'method': self._method,
            'url': self._url,
            'headers': self._headers,
            'timeout': self._request_timeout_secs,

            # when self._query_params_fn this will be overwritten on each iteration
            'params': self._query_params
        }

        request_args.update(self._requst_kwargs)

        while (self._stop_after == 0 or num_messages < self._stop_after):
            if self._query_params_fn is not None:
                request_args['params'] = self._query_params_fn()

            (http_session, df) = http_utils.request(request_args,
                                                    requests_session=http_session,
                                                    max_retries=self._max_retries,
                                                    sleep_time=self._error_sleep_time,
                                                    respect_retry_after_header=self._respect_retry_after_header,
                                                    accept_status_codes=self._accept_status_codes,
                                                    on_success_fn=self._parse_response)

            # Even if we didn't receive any errors, the server may not have had any data for us.
            if df is not None and len(df):
                num_messages += len(df)
                yield MessageMeta(df)

            time.sleep(self._sleep_time)

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node, MessageMeta
