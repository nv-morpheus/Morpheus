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


@register_stage("to-rest", ignore_args=["query_params", "headers", "**request_kwargs"])
class WriteToRestStage(SinglePortStage):
    """
    Write all messages to a Kafka cluster.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    base_url : str
        Server base url, should include the intended protocol (e.g. http:// or https://) and port if necessary.
        This may or may not include a base path from which `endpoint` will be appended.
        examples:
            "https://nvcr.io/"
            "http://localhost:8080/base/path"
    endpoint : str
        Endpoint to which messages will be sent. This will be appended to `base_url` and may include a query string.

        When `static_endpoint` is `False` this may contain named format strings which will be replaced with the
        corresponding column value from the first row of the incoming dataframe, if no such column exists a `ValueError`
         will be raised.
        examples:
            "api/v1/endpoint"
            "api/v1/endpoint?time={timestamp}&id={id}"
            "/{model_name}/{user}?time={timestamp}"
    static_endpoint : bool, default True
        Setting this to `True` indicates that the value of `endpoint` does not change between requests, and can be an
        optimization.
    method : str, optional
        HTTP method to use when sending messages, by default "POST". Currently only "POST", "PUT" and "PATCH" are
        supported.
    headers : dict, optional
        Optional set of headers to include in the request.
        If `None` the header value will be inferred based on `lines`.
            * `{"Content-Type": "text/plain"}` when `lines` is `True`
            * `{"Content-Type": "application/json"}` when `lines` is `False`
    accept_status_codes :  typing.List[int][int], optional,  multiple = True
        List of acceptable status codes, by default (200, 201, 202).

    max_rows_per_payload : int, optional
        Maximum number of rows to include in a single payload, by default 10000.
        Setting this to 1 will send each row as a separate request.
    lines : bool, default False
        If False, dataframes will be serialized to a JSON array of objects. If True, then the dataframes will be
        serialized to a string JSON objects separated by end-of-line characters.
    **request_kwargs : dict
        Additional arguments to pass to the `requests.request` function.
    """

    def __init__(self,
                 c: Config,
                 base_url: str,
                 endpoint: str,
                 static_endpoint: bool = True,
                 headers: dict = None,
                 query_params: dict = None,
                 method: str = "POST",
                 error_sleep_time: float = 0.1,
                 respect_retry_after_header: bool = True,
                 request_timeout_secs: int = 30,
                 accept_status_codes: typing.List[int] = (
                     200,
                     201,
                     202,
                 ),
                 max_retries: int = 10,
                 max_rows_per_payload: int = 10000,
                 lines: bool = False,
                 df_to_request_kwargs_fn: typing.Callable[[DataFrameType], dict] = None,
                 **request_kwargs):
        super().__init__(c)
        self._base_url = http_utils.verify_url(base_url)

        if (callable(endpoint) and static_endpoint):
            raise ValueError("endpoint must be a string when static_endpoint is True")

        self._endpoint = endpoint
        self._query_params = query_params

        if headers is None:
            if lines:
                headers = {"Content-Type": "text/plain"}
            else:
                headers = {"Content-Type": "application/json"}

        self._headers = headers

        self._method = method

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
        self._static_endpoint = static_endpoint
        self._max_rows_per_payload = max_rows_per_payload
        self._lines = lines
        self._df_to_request_kwargs_fn = df_to_request_kwargs_fn
        self._requst_kwargs = request_kwargs
        self._http_session = None

    @property
    def name(self) -> str:
        return "to-rest"

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

    def _df_to_url(self, df: DataFrameType) -> str:
        """
        Convert a Dataframe to a URL. Only called when self._static_endpoint is False.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to infer enpoint from.

        Returns
        -------
        str
            URL.
        """
        endpoint = self._endpoint.format(df.iloc[0].to_dict())
        return f"{self._base_url}{endpoint}"

    def _df_to_payload(self, df: DataFrameType) -> StringIO:
        str_buf = StringIO()
        serializers.df_to_stream_json(df=df, stream=str_buf, lines=self._lines)
        str_buf.seek(0)
        return str_buf

    def _chunk_requests(self, df: DataFrameType) -> typing.Iterable[dict]:
        """
        Convert a Dataframe to a series of urls and payloads with no more than `self._max_rows_per_payload`.

        Parameters
        ----------
        df : DataFrameType
            DataFrame to chunk, and convert to urls and payloads.
        """
        slice_start = 0
        while (slice_start < len(df)):
            slice_end = min(slice_start + self._max_rows_per_payload, len(df))
            df_slice = df.iloc[slice_start:slice_end]

            if self._df_to_request_kwargs_fn is not None:
                yield self._df_to_request_kwargs_fn(df_slice)
            else:
                chunk = {'payload': self.df_to_payload(df_slice)}
                if not self._static_endpoint:
                    chunk['url'] = self.df_to_url(df_slice)

                yield chunk

            slice_start = slice_end

    def _process_message(self, msg: MessageMeta) -> MessageMeta:

        request_args = {
            'method': self._method,
            'headers': self._headers,
            'timeout': self._request_timeout_secs,
            'params': self._query_params
        }

        if self._static_endpoint:
            request_args['url'] = f"{self._base_url}{self._endpoint}"

        request_args.update(self._requst_kwargs)

        for chunk in self._chunk_requests(msg.df):
            request_args.update(chunk)
            http_utils.request(request_args,
                               requests_session=self._http_session,
                               max_retries=self._max_retries,
                               sleep_time=self._error_sleep_time,
                               respect_retry_after_header=self._respect_retry_after_header,
                               accept_status_codes=self._accept_status_codes)

        return msg

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
