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

import typing

import cupy as cp
import mrc
import pandas as pd

import cudf

from morpheus._lib.common import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import ResponseMemory
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils import compare_df
from morpheus.utils.atomic_integer import AtomicInteger


@register_stage("unittest-conv-msg")
class ConvMsg(SinglePortStage):
    """
    Simple test stage to convert a MultiMessage to a MultiResponseProbsMessage
    Basically a cheap replacement for running an inference stage.

    Setting `expected_data_file` to the path of a cav/json file will cause the probs array to be read from file.
    Setting `expected_data_file` to `None` causes the probs array to be a copy of the incoming dataframe.
    Setting `columns` restricts the columns copied into probs to just the ones specified.
    Setting `order` specifies probs to be in either column or row major
    Setting `empty_probs` will create an empty probs array with 3 columns, and the same number of rows as the dataframe
    """

    def __init__(self,
                 c: Config,
                 expected_data_file: str = None,
                 columns: typing.List[str] = None,
                 order: str = 'K',
                 probs_type: str = 'f4',
                 empty_probs: bool = False):
        super().__init__(c)
        self._expected_data_file = expected_data_file
        self._columns = columns
        self._order = order
        self._probs_type = probs_type
        self._empty_probs = empty_probs

    @property
    def name(self):
        return "test"

    def accepted_types(self):
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    def _conv_message(self, m):
        if self._expected_data_file is not None:
            df = read_file_to_df(self._expected_data_file, FileTypes.CSV, df_type="cudf")
        else:
            if self._columns is not None:
                df = m.get_meta(self._columns)
            else:
                df = m.get_meta()

        if self._empty_probs:
            probs = cp.zeros([len(df), 3], 'float')
        else:
            probs = cp.array(df.values, dtype=self._probs_type, copy=True, order=self._order)

        memory = ResponseMemory(count=len(probs), tensors={'probs': probs})
        return MultiResponseMessage.from_message(m, memory=memory)

    def _build_single(self, builder: mrc.Builder, input_stream):
        stream = builder.make_node(self.unique_name, self._conv_message)
        builder.make_edge(input_stream[0], stream)

        return stream, MultiResponseMessage


@register_stage("unittest-dfp-length-check")
class DFPLengthChecker(SinglePortStage):
    """
    Verifies that the incoming MessageMeta classes are of a specific length

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    expected_length : int
        Expected length of incoming messages

    num_exact: int
        Number of messages to check. For a datasource with 2000 records if we expect the first message to be of lenth
        1024, the next message will contain only 976. The first `num_exact` messages will be checked with `==` and `<=`
        after that.
    """

    def __init__(self, c: Config, expected_length: int, num_exact: int = 1):
        super().__init__(c)

        self._expected_length = expected_length
        self._num_exact = num_exact
        self._num_received = AtomicInteger(0)

    @property
    def name(self) -> str:
        return "dfp-length-check"

    def accepted_types(self) -> typing.Tuple:
        return (MessageMeta)

    def supports_cpp_node(self):
        return False

    def _length_checker(self, x: MessageMeta):
        msg_num = self._num_received.get_and_inc()
        if msg_num < self._num_exact:
            assert x.count == self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} != {self._num_exact}"
        else:
            assert x.count <= self._expected_length, \
                f"Unexpected number of rows in message number {msg_num}: {x.count} > {self._num_exact}"

        return x

    def _build_single(self, builder: mrc.Builder, input_stream):
        node = builder.make_node(self.unique_name, self._length_checker)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]

