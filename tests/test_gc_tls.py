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
import mrc.core.operators as ops
import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.type_aliases import DataFrameType


class FftStage(SinglePortStage):
    """
    Simple stage that performs FFT calculation
    """

    @property
    def name(self) -> str:
        return "fft_stage"

    def supports_cpp_node(self) -> bool:
        return False

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def on_next(self, x: typing.Any):
        cp.fft.fft(cp.zeros(10))
        return x

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self.on_next))
        builder.make_edge(input_stream[0], node)
        return node, input_stream[1]


def run_pipe2(df: DataFrameType):
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, dataframes=[df]))
    pipe.add_stage(FftStage(config))
    pipe.run()


def run_pipe1(df: DataFrameType):
    """
    Simple C++ pipeline where the sink holds on to a reference to the message
    """
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(InMemorySinkStage(config))
    pipe.run()


@pytest.mark.use_cpp
@pytest.mark.usefixtures("disable_gc")
def test_gc_tls(filter_probs_df: DataFrameType):
    """
    Test for MRC issue #362 where `gc.collect()` is invoked while a thread is being finalized and thus
    thread-local storage has already been cleared, when this happens and `gc.collect()` collects an object who's
    destructor makes use of pybind11's `gil_scoped_acquire` then an internal pybind11 error is triggered.

    Specifically the destructor for `PyDataTable` makes use of `gil_scoped_acquire` to release the GIL to allow the
    underlying dataframe to be freed by the python interpreter.

    The easiest was to reproduce this is to run a pipeline with C++ mode enabled wehre the sink like
    `InMemorySinkStage` holds on to a reference to the message not allowing it to be garbage collected until after the
    pipeline has finished running. Then run a second pipeline in either C++ or Python mode, where once of the stages
    has a finalizer on the thread which calls `gc.collect` specifically `cupy.fft` does this and I was unable to repro
    the issue myself, possibly since that code was written in cython.

    This only works if the first pipeline and thus the `PyDataTable` are not gargabe collected before the second
    pipeline is collected which makes this difficult to reproduce, but it is possible to force the issue by disabling
    automatic garbage collection.
    """
    run_pipe1(filter_probs_df)
    run_pipe2(filter_probs_df)
