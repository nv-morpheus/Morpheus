#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import morpheus.modules  # noqa: F401
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.multi_inputport_modules_stage import MultiInputportModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import MULTIPLEXER
from utils.dataset_manager import DatasetManager


def test_multi_input_pipeline(config, dataset_cudf: DatasetManager):
    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    # Create the stages
    source_stage_1 = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
    source_stage_2 = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))

    multiplexer_module_conf = {
        "module_id": MULTIPLEXER,
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "module_name": "multiplexer",
        "num_input_ports_to_merge": 2,
        "streaming": False
    }

    mux_stage = pipe.add_stage(MultiInputportModulesStage(config, multiplexer_module_conf, num_input_ports_to_merge=2))

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    # Create the edges
    pipe.add_edge(source_stage_1, mux_stage.input_ports[0])
    pipe.add_edge(source_stage_2, mux_stage.input_ports[1])
    pipe.add_edge(mux_stage, sink_stage)

    pipe.run()

    # We should expect to see two messages in the sink as two sources have generated one message each.
    assert len(sink_stage.get_messages()) == 2


def test_multi_input_pipeline2(config, dataset_cudf: DatasetManager):
    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    # Create the stages
    source_stage_1 = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
    source_stage_2 = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
    source_stage_3 = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))

    multiplexer_module_conf = {
        "module_id": MULTIPLEXER,
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "module_name": "multiplexer",
        "num_input_ports_to_merge": 3,
        "streaming": False
    }

    mux_stage = pipe.add_stage(MultiInputportModulesStage(config, multiplexer_module_conf, num_input_ports_to_merge=3))

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    # Create the edges
    pipe.add_edge(source_stage_1, mux_stage.input_ports[0])
    pipe.add_edge(source_stage_2, mux_stage.input_ports[1])
    pipe.add_edge(source_stage_3, mux_stage.input_ports[2])
    pipe.add_edge(mux_stage, sink_stage)

    pipe.run()

    # We should expect to see three messages in the sink as three sources have generated one message each.
    assert len(sink_stage.get_messages()) == 3
