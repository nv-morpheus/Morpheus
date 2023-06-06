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
import requests
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils import http_utils

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
        Endpoint to which messages will be sent. This will be appended to `base_url` and may include a query string and
        named format strings. Named format strings will be replaced with the corresponding column value from the first
        row of the incoming dataframe, if no such column exists a `ValueError` will be raised.
        examples:
            "api/v1/endpoint"
            "api/v1/endpoint?time={timestamp}&id={id}"
            "/{model_name}/{user}?time={timestamp}"
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

    <Probably need some timeout and retry type args here>
    """

    def __init__(self,
                 c: Config,
                 base_url: str,
                 endpoint: str,
                 headers: dict = None,
                 query_params: typing.Union[dict, typing.Callable] = None,
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
                 lines: bool = False,
                 static_endpoint: bool = True,
                 **request_kwargs):
        super().__init__(c)
        self._base_url = http_utils.verify_url(base_url)
        self._endpoint = endpoint

        if callable(query_params):
            self._query_params_fn = query_params
            self._query_params = None
        else:
            self._query_params_fn = None
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
        self._lines = lines
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

    def msg_to_url(self, x: MessageMeta) -> str:
        """
        Convert a message to a URL.

        Parameters
        ----------
        x : `morpheus.messages.MessageMeta`
            Message to convert.

        Returns
        -------
        str
            URL.

        """
        if self._static_endpoint:
            endpoint = self._endpoint
        else:
            endpoint = self._endpoint.format(**x.df.iloc[0].to_dict())

        return f"{self._base_url}{endpoint}"

    def msg_to_payloads(self, msg: MessageMeta) -> typing.List[StringIO]:
        """
        Convert a message to a payload.

        Parameters
        ----------
        msg : `morpheus.messages.MessageMeta`
            Message to convert.

        Returns
        -------
        StringIO
            Payload.

        """
        str_buf = StringIO()
        serializers.df_to_stream_json(df=msg.df, stream=str_buf, lines=self._lines)
        str_buf.seek(0)

        # TODO apply chunking based on byte size of str_buf
        return [str_buf]

    def _process_message(self, msg: MessageMeta) -> MessageMeta:
        url = self.msg_to_url(msg)
        payloads = self.msg_to_payloads(msg)

        request_args = {
            'method': self._method,
            'url': url,
            'headers': self._headers,
            'timeout': self._request_timeout_secs,
            'params': self._query_params
        }

        if self._query_params_fn is not None:
            request_args['params'] = self._query_params_fn()

        request_args.update(self._requst_kwargs)

        for payload in payloads:
            request_args['data'] = payload
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
