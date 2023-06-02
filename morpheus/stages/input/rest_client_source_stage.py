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

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"Content-Type": "application/json"}


@register_stage("from-rest-client", ignore_args=["query_params", "headers", "**request_kwargs"])
class RestClientSourceStage(PreallocatorMixin, SingleOutputSource):
    """
    Source stage that polls a remote HTTP server for incoming data.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    url : str
        Remote URL to poll for data, ex `https://catalog.ngc.nvidia.com/api/collections`.
    query_params : dict, callable, default None
        Query parameters to pass to the remote URL. Can either be a dictionary of key-value pairs or a callable that
        returns a dictionary of key-value pairs. If a callable is provided, it will be called with no arguments.
    headers: dict, default None
        Headers sent with the request. If `None` then `{"Content-Type": "application/json"}` will be used.
    method : str, default "GET"
        HTTP method to use.
    sleep_time : float, default 0.1
        Amount of time in seconds to sleep between successive requests. Setting this to 0 disables this feature.
    error_sleep_time : float, default 0.1
        Amount of time in seconds to sleep after the client receives an error.
        The client will perform an exponential backoff starting at `error_sleep_time`.
        Setting this to 0 causes the client to poll the remote server as fast as possible.
        If the server sets a `Retry-After` header, then that value will take precedence over `error_sleep_time`.
    max_errors : int, default 10
        Maximum number of consequtive errors to receive before raising an error.
    accept_status_codes : typing.List[int], optional,  multiple = True
        List of status codes to accept. If the response status code is not in this tuple, then the request will be
        considered an error
    max_retries : int, default 10
        Maximum number of times to retry the request fails, receives a redirect or returns a status in the
        `retry_status_codes` list. Setting this to 0 disables this feature, and setting this to a negative number will raise
        a `ValueError`.
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
                 request_timeout_secs: int = 30,
                 accept_status_codes: typing.List[int] = (200, ),
                 max_retries: int = 10,
                 lines: bool = False,
                 **request_kwargs):
        super().__init__(config)
        self._url = url

        if callable(query_params):
            self._query_params_fn = query_params
        else:
            if query_params is None:
                query_params = {}

            self._query_params_fn = lambda: query_params

        self._headers = headers or DEFAULT_HEADERS.copy()
        self._method = method

        if sleep_time >= 0:
            self._sleep_time = sleep_time
        else:
            raise ValueError("sleep_time must be >= 0")

        if error_sleep_time >= 0:
            self._error_sleep_time = error_sleep_time
        else:
            raise ValueError("error_sleep_time must be >= 0")

        self._request_timeout_secs = request_timeout_secs
        if max_retries >= 0:
            self._max_retries = max_retries
        else:
            raise ValueError("max_retries must be >= 0")

        self._accept_status_codes = tuple(accept_status_codes)

        self._lines = lines
        self._requst_kwargs = request_kwargs

    @property
    def name(self) -> str:
        return "from-rest-client"

    def supports_cpp_node(self) -> bool:
        return False

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        # Manully implementing retry logic, urllib3 includes retry logic, however there is a known issue preventing us
        # from using it: https://github.com/urllib3/urllib3/issues/2751

        num_errors = 0  # Number of consequtive errors received, any successful responses resets this number

        # Not using session as a context manager, allowing us to reconstruct it on a connection error
        http_session = None
        while (num_errors <= self._max_retries):
            if http_session is None:
                http_session = requests.Session()

            has_error = False  # Set true on a raised exception or non-accepted status code
            retry_after = None  # Set to a number of seconds to sleep if the server sent us a retry hint
            payload = None
            try:
                response = http_session.request(self._method,
                                                self._url,
                                                params=self._query_params_fn(),
                                                headers=self._headers,
                                                timeout=self._request_timeout_secs,
                                                **self._requst_kwargs)
                if response.status_code in self._accept_status_codes:
                    payload = response.content
                else:
                    logger.error("Received unexpected status code %d: %s", response.status_code, response.text)
                    has_error = True

                    # often set for 429 and 503 responses, for other statuses it simply won't be set
                    if 'Retry-After' in response.headers:
                        try:
                            retry_after = int(response.headers['Retry-After'])
                        except Exception as e:
                            logger.error("Error occurred parsing Retry-After header: %s", e)
                            retry_after = None

            except requests.exceptions.RequestException:
                logger.error("Error occurred requesting data from %s", self._url)
                try:
                    http_session.close()
                except:
                    pass

                http_session = None
                has_error = True

            df = None
            if not has_error and len(payload) > 2:
                try:
                    df = cudf.read_json(payload, lines=self._lines, engine='cudf')
                except Exception as e:
                    logger.error("Error occurred converting response payload to Dataframe: %s", e)
                    has_error = True

            if not has_error:
                # Reset retry logic
                num_errors = 0

                # Even if we didn't receive any errors, we may not have received any data
                if df is not None and len(df):
                    yield MessageMeta(df)
            else:
                num_errors += 1

            if num_errors <= self._max_retries:
                # Only sleep if we aren't exiting
                if not has_error:
                    sleep_time = self._sleep_time
                else:
                    # Respect the Retry-After header if it was set
                    if retry_after is not None:
                        sleep_time = retry_after
                    else:
                        sleep_time = (2**(num_errors - 1)) * self._error_sleep_time

                logger.debug("Sleeping for %s seconds before polling again", self._sleep_time)
                time.sleep(sleep_time)

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node, MessageMeta
