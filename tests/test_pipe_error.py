#!/usr/bin/env python
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
import typing

import mrc
import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.logger import configure_logging
from utils.stages.error_raiser import ErrorRaiserStage


class InMemSourceXStage(SingleOutputSource):
    """
    InMemorySourceStage subclass that emits whatever you give it and doesn't assume the source data
    is a dataframe.
    """

    def __init__(self, c: Config, data: typing.List[typing.Any]):
        super().__init__(c)
        self._data = data

    @property
    def name(self) -> str:
        return "from-data"

    def supports_cpp_node(self) -> bool:
        return False

    def _emit_data(self) -> typing.Iterator[typing.Any]:
        for x in self._data:
            yield x

    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        node = builder.make_source(self.unique_name, self._emit_data())
        return node, type(self._data[0])


@pytest.mark.parametrize("exception_cls", [RuntimeError, ValueError, NotImplementedError])
def test_stage_raises_exception(config: Config, filter_probs_df: pd.DataFrame, exception_cls: type[Exception]):
    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    error_raiser_stage = pipe.add_stage(ErrorRaiserStage(config, exception_cls=exception_cls))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    with pytest.raises(exception_cls):
        pipe.run()

    # Ensure that the raised exception was from our stage and not from something else
    assert error_raiser_stage.error_raised
    assert len(sink_stage.get_messages()) == 0


@pytest.mark.use_python
@pytest.mark.parametrize("delayed_start", [False, True])
def test_monitor_not_impl(config: Config, delayed_start: bool):

    class UnsupportedType:
        pass

    pipe = LinearPipeline(config)
    pipe.set_source(InMemSourceXStage(config, [UnsupportedType()]))
    monitor_stage = pipe.add_stage(MonitorStage(config, log_level=logging.WARNING, delayed_start=delayed_start))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    assert monitor_stage._mc.is_enabled()

    with pytest.raises(NotImplementedError):
        pipe.run()

    assert len(sink_stage.get_messages()) == 0
