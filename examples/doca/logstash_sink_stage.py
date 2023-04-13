# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import typing
import socket
import json
import sys

from transports import TcpTransport
import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io import serializers
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


@register_stage("to-logstash")
class WriteToLogStashStage(SinglePortStage):
    """
    Write all messages to a logstash pipeline cluster.
    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    host : str
        Logstash IP address
    port : int
        Logstash TCP port
    """

    def __init__(self, c: Config, host: str, port: int, timeout: int = 10):
        super().__init__(c)
        
        self._timeout = timeout
                
        self._transport = TcpTransport(
            host=host, 
            port=port,
            timeout=10000)

    @property
    def name(self) -> str:
        return "to-logstash"

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

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Convert the messages to rows of strings
        stream = input_stream[0]

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            def on_data(x: MessageMeta):
            
                records = serializers.df_to_json(x.df, strip_newlines=True)
                
                for m in records:
                    self._transport.send([m])
  
                return x

            obs.pipe(ops.map(on_data)).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)
        # cpu cores
        node.launch_options.pe_count = 1
        # threads per core
        node.launch_options.engines_per_pe = self._num_threads

        builder.make_edge(stream, node)

        return node, input_stream[1]

