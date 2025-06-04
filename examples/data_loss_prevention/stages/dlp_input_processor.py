# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.control_message_stage import ControlMessageStage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.utils.type_aliases import SeriesType
from morpheus.utils.type_utils import get_df_class


@register_stage("dlp_input_processor", modes=[PipelineModes.NLP])
class DLPInputProcessor(ControlMessageStage, GpuAndCpuMixin):
    """Handles input text processing and normalization for DLP pipeline"""

    def __init__(self,
                 config: Config,
                 *,
                 column_name: str = "source_text",
                 chunking_size: int = 1000,
                 split_by_paragraphs: bool = False):
        super().__init__(config)
        self.column_name = column_name
        self.chunking_size = chunking_size
        self.split_by_paragraphs = split_by_paragraphs
        self.df_class = get_df_class(config.execution_mode)

    @property
    def name(self) -> str:
        return "dlp_input_processor"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        # Enable support by default
        return False

    def preprocess(self, msg: MessageMeta) -> ControlMessage:
        """
        Preprocess input text:
        1. Normalize whitespace
        2. Split into manageable chunks for processing
        """

        new_rows = []
        with msg.mutable_dataframe() as df:
            source_series: SeriesType = df[self.column_name]

            if not isinstance(source_series, pd.Series):
                # cudf series doesn't support iteration
                source_series = source_series.to_arrow().to_pylist()

            for row in source_series:
                # Basic normalization
                normalized_text = row.replace('\r\n', '\n').replace('\r', '\n')

                # For larger texts, split into chunks to optimize processing
                if self.split_by_paragraphs:
                    # Split by paragraphs first to preserve content boundaries
                    paragraphs = normalized_text.split('\n\n')
                    current_chunk = []
                    current_chunk_len = 0

                    for para in paragraphs:
                        if current_chunk_len > 0 and (current_chunk_len + len(para) > self.chunking_size):
                            new_rows.append("".join(current_chunk))
                            current_chunk = [para]
                            current_chunk_len = len(para)
                        else:
                            if len(current_chunk) > 0:
                                current_chunk.append("\n\n")
                                current_chunk_len += 2

                            current_chunk.append(para)
                            current_chunk_len += len(para)

                    if len(current_chunk):
                        new_rows.append("".join(current_chunk))

                else:
                    new_rows.append(normalized_text)

        new_df = self.df_class({self.column_name: new_rows})
        control_msg = ControlMessage()
        control_msg.payload(MessageMeta(new_df))

        return control_msg

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.preprocess))
        builder.make_edge(input_node, node)

        return node
