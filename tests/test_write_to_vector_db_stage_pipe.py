#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.modules import to_control_message  # noqa: F401 # pylint: disable=unused-import
from morpheus.pipeline import LinearPipeline
from morpheus.service.milvus_vector_db_service import MilvusVectorDBService
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.write_to_vector_db import WriteToVectorDBStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE


@pytest.mark.use_cpp
def test_write_to_vector_db_stage_with_instance_pipe(config: Config, pipeline_batch_size: int = 256):
    config.pipeline_batch_size = pipeline_batch_size

    rows_count = 5
    dimensions = 10
    df = cudf.DataFrame({
            "id": [i for i in range(rows_count)],
            "age": [random.randint(20, 40) for i in range(rows_count)],
            "embedding": [[random.random() for _ in range(dimensions)] for _ in range(rows_count)]
        })

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE,
        "module_name": "to_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE
    }
    vdb_service = MilvusVectorDBService(uri="http://localhost:19530")

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(LinearModulesStage(config, to_cm_module_config,
                                      input_port_name="input",
                                      output_port_name="output",
                                      output_type=ControlMessage))
    pipe.add_stage(WriteToVectorDBStage(config, resource_name="test", vdb_service=vdb_service))

    pipe.run()


@pytest.mark.use_cpp
def test_write_to_vector_db_stage_with_name_pipe(config: Config, pipeline_batch_size: int = 256):
    config.pipeline_batch_size = pipeline_batch_size

    rows_count = 5
    dimensions = 10
    df = cudf.DataFrame({
            "id": [i for i in range(rows_count)],
            "age": [random.randint(20, 40) for i in range(rows_count)],
            "embedding": [[random.random() for _ in range(dimensions)] for _ in range(rows_count)]
        })

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE,
        "module_name": "to_control_message",
        "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(LinearModulesStage(config, to_cm_module_config,
                                      input_port_name="input",
                                      output_port_name="output",
                                      output_type=ControlMessage))
    pipe.add_stage(WriteToVectorDBStage(config,
                                        resource_name="test_collection",
                                        vdb_service="milvus",
                                        uri="http://localhost:19530"))

    pipe.run()
