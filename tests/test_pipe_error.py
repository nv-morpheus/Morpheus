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

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from utils.stages.error_raiser import ErrorRaiserStage


@pytest.mark.parametrize("num_threads", [1, 2, 8])
def test_stage_raises_exception(config: Config, filter_probs_df: pd.DataFrame, num_threads: int):
    config.num_threads = num_threads

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(ErrorRaiserStage(config))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    with pytest.raises(RuntimeError):
        pipe.run()

    assert len(sink_stage.get_messages()) == 0
