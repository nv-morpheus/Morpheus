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
from urllib3.util import Retry

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"Content-Type": "application/json"}
DEFAULT_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


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
    retry_status_codes: typing.List[int], optional, multiple = True
        List of status codes to retry if the request fails. If `None`, then the `DEFAULT_RETRY_STATUS_CODES` list
        of status codes is used. Raises a `ValueError` if there is any ovwerlap between `accept_status_codes` and
        `retry_status_codes`. Setting this to an empty list disables this feature, and retries will only be performed
        for network errors. If the client receives a status code not in `accept_status_codes` or `retry_status_codes`,
        then the error will be considered to be fatal.
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
                 retry_status_codes: typing.List[int] = DEFAULT_RETRY_STATUS_CODES,
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

        if len(set(accept_status_codes) & set(retry_status_codes)) > 0:
            raise ValueError("accept_status_codes and retry_status_codes must not overlap")

        self._accept_status_codes = tuple(accept_status_codes)
        self._retry_status_codes = tuple(retry_status_codes)

        self._lines = lines
        self._requst_kwargs = request_kwargs

    @property
    def name(self) -> str:
        return "from-rest-client"

    def supports_cpp_node(self) -> bool:
        return False

    def _generate_frames(self) -> typing.Iterator[MessageMeta]:
        with requests.Session() as http_session:
            if self._max_retries > 0:
                # https://urllib3.readthedocs.io/en/1.26.15/reference/urllib3.util.html#urllib3.util.Retry
                retry = Retry(total=self._max_retries,
                              backoff_factor=self._error_sleep_time,
                              respect_retry_after_header=True,
                              raise_on_redirect=True,
                              raise_on_status=True,
                              status_forcelist=self._retry_status_codes)
                retry_adapter = requests.adapters.HTTPAdapter(max_retries=retry)
                http_session.mount(self._url, retry_adapter)

            fatal_error = False
            while (not fatal_error):
                payload = None
                try:
                    # Known issue: https://github.com/urllib3/urllib3/issues/2751
                    # If the connection to remote goes down during the request (not before), then an exception will be
                    # raised immediately bypassing the retry logic.
                    # TODO: Revert back to home-grown retry logic.
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
                        fatal_error = True

                except requests.exceptions.RequestException:
                    logger.error("Error occurred requesting data from %s", self._url)
                    fatal_error = True

                df = None
                if not fatal_error and len(payload) > 2:
                    # Work-around for https://github.com/rapidsai/cudf/issues/5712
                    try:
                        df = cudf.read_json(payload, lines=self._lines, engine='cudf')
                    except Exception as e:
                        logger.error("Error occurred converting response payload to Dataframe: %s", e)

                if df is not None and len(df):
                    yield MessageMeta(df)

                if not fatal_error:
                    # We didn't encounter an error, however the server didn't have any new data for us
                    logger.debug("Sleeping for %s seconds before polling again", self._sleep_time)
                    time.sleep(self._sleep_time)

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._generate_frames())
        return node, MessageMeta
