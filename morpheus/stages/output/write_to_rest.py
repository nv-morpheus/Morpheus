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
import time
import typing

import confluent_kafka as ck
import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("to-rest")
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
    acceptable_status_codes : typing.Tuple[int], optional
        Tuple of acceptable status codes, by default (200, 201, 202).

    <Probably need some timeout and retry type args here>
    """

    def __init__(self,
                 c: Config,
                 base_url: str,
                 endpoint: str,
                 method: str = "POST",
                 headers: dict = None,
                 acceptable_status_codes: typing.Tuple[int] = (
                     200,
                     201,
                     202,
                 )):
        super().__init__(c)
        self._base_url = base_url
        self._endpoint = endpoint
        self._method = method
        self._headers = headers or {}

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
        endpoint = self._endpoint.format(**x.df.iloc[0].to_dict())
        return f"{self._base_url}/{endpoint}"

    def msg_to_payload(self, x: MessageMeta) -> typing.List[str]:
        """
        Convert a message to a payload.

        Parameters
        ----------
        x : `morpheus.messages.MessageMeta`
            Message to convert.

        Returns
        -------
        typing.List[str]
            Payload.

        """
        return serializers.df_to_json(x.df, strip_newlines=True)

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

    def _process_message(self, msg: MessageMeta) -> MessageMeta:
        return msg

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
