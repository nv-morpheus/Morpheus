# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import inspect
import logging
import typing
from abc import abstractmethod
from functools import partial

import cupy as cp
import neo
import numpy as np
import pandas as pd
import typing_utils
from neo.core import operators as ops

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import InferenceMemoryFIL
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiInferenceMessage
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiMessage
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.cudf_subword_helper import tokenize_text_series


class DropNullStage(SinglePortStage):
    """
    Drop null/empty data input entries.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    column : str
        Column name to perform null check.

    """

    def __init__(self, c: Config, column: str):
        super().__init__(c)

        self._column = column

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "dropna"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (MessageMeta, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        # Finally, flatten to a single stream
        def node_fn(input: neo.Observable, output: neo.Subscriber):

            def on_next(x: MessageMeta):

                y = MessageMeta(x.df[~x.df[self._column].isna()])

                return y

            input.pipe(ops.map(on_next), ops.filter(lambda x: not x.df.empty)).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node

        return stream, input_stream[1]
