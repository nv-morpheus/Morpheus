# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import typing

import mrc

from morpheus._lib.type_id import TypeId
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("recipient-features", modes=[PipelineModes.NLP])
class RecipientFeaturesStage(SinglePortStage):
    """
    Pre-processing stage which counts the number of recipients in an email's metadata.

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    sep_token : str
        Bert separator token.
    """

    def __init__(self, config: Config, sep_token: str = '[SEP]'):
        super().__init__(config)
        if config.mode != PipelineModes.NLP:
            raise RuntimeError("RecipientFeaturesStage must be used in a pipeline configured for NLP")

        if len(sep_token):
            self._sep_token = sep_token
        else:
            raise ValueError("sep_token cannot be an empty string")

        self._needed_columns.update({
            'to_count': TypeId.INT32,
            'bcc_count': TypeId.INT32,
            'cc_count': TypeId.INT32,
            'total_recipients': TypeId.INT32,
            'data': TypeId.STRING
        })

    @property
    def name(self) -> str:
        return "recipient-features"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: MessageMeta) -> MessageMeta:
        # Open the DataFrame from the incoming message for in-place modification
        with message.mutable_dataframe() as ctx:
            ctx.df['to_count'] = ctx.df['To'].str.count('@')
            ctx.df['bcc_count'] = ctx.df['BCC'].str.count('@')
            ctx.df['cc_count'] = ctx.df['CC'].str.count('@')
            ctx.df['total_recipients'] = ctx.df['to_count'] + ctx.df['bcc_count'] + ctx.df['cc_count']

            # Attach features to string data
            ctx.df['data'] = (ctx.df['to_count'].astype(str) + '[SEP]' + ctx.df['bcc_count'].astype(str) + '[SEP]' +
                              ctx.df['cc_count'].astype(str) + '[SEP]' + ctx.df['total_recipients'].astype(str) +
                              '[SEP]' + ctx.df['Message'])

        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
