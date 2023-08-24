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

import pytest

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import modules.multiplexer  # noqa: F401 # pylint: disable=unused-import
from _utils.dataset_manager import DatasetManager
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.multi_port_modules_stage import MultiPortModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


@pytest.mark.parametrize("source_count, expected_count", [(1, 1), (2, 2), (3, 3)])
def test_multi_port_pipeline(config, dataset_cudf: DatasetManager, source_count, expected_count):

    filter_probs_df = dataset_cudf["filter_probs.csv"]

    pipe = Pipeline(config)

    input_ports = []
    for x in range(source_count):
        input_port = f"input_{x}"
        input_ports.append(input_port)

    multiplexer_module_conf = {
        "module_id": "multiplexer",
        "namespace": "morpheus_test",
        "module_name": "multiplexer",
        "input_ports": input_ports,
        "output_port": "output"
    }

    mux_stage = pipe.add_stage(
        MultiPortModulesStage(config, multiplexer_module_conf, input_ports=input_ports, output_ports=["output"]))

    for x in range(source_count):
        source_stage = pipe.add_stage(InMemorySourceStage(config, [filter_probs_df]))
        pipe.add_edge(source_stage, mux_stage.input_ports[x])

    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(mux_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count
