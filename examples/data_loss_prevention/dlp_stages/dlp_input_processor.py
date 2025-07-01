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

import logging
import time
import typing

import mrc
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

logger = logging.getLogger(f"morpheus.{__name__}")


@register_stage("dlp_input_processor", modes=[PipelineModes.NLP])
class DLPInputProcessor(GpuAndCpuMixin, ControlMessageStage):
    """
    Handles input text processing and normalization for DLP pipeline

    Parameters
    ----------
    config : morpheus.config.Config
        Pipeline configuration instance.
    column_name : str
        Name of the column containing the source text to process.
    split_paragraphs : bool
        If True, splits input text into chunks. Defaults to False.
    """

    def __init__(self,
                 config: Config,
                 *,
                 column_name: str = "source_text",
                 split_paragraphs: bool = True,
                 log_elapsed_time: bool = False):
        super().__init__(config)
        self.column_name = column_name
        self.split_paragraphs = split_paragraphs
        self.df_class = get_df_class(config.execution_mode)
        self._elapsed_time = 0.0
        self._log_elapsed_time = log_elapsed_time

    @property
    def name(self) -> str:
        return "dlp_input_processor"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (ControlMessage, )

    def supports_cpp_node(self):
        # Enable support by default
        return False

    def preprocess(self, msg: ControlMessage) -> ControlMessage:
        """
        Preprocess input text:
        1. Normalize whitespace
        2. Split into manageable chunks for processing
        """

        t1 = time.time()
        with msg.payload().mutable_dataframe() as df:
            if df.index.name is None:
                df.index.name = "index"

            source_series: SeriesType = df[self.column_name]
            source_series = source_series.str.replace('\r\n', '\n').str.replace('\r', '\n')
            if not self.split_paragraphs:
                df[self.column_name] = source_series
            else:
                split_series = source_series.str.split("\n").explode()
                new_df = self.df_class({self.column_name: split_series})

                new_df.index.name = "original_source_index"
                new_df.reset_index(drop=False, inplace=True)

                meta = MessageMeta(new_df)
                msg.payload(meta)

        t2 = time.time()
        self._elapsed_time += (t2 - t1)
        return msg

    def log_elapsed_time(self):
        if self._log_elapsed_time:
            logger.warning("Elapsed time for %s: %.4f seconds", self.name, self._elapsed_time)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.preprocess), ops.on_completed(self.log_elapsed_time))
        builder.make_edge(input_node, node)

        return node
