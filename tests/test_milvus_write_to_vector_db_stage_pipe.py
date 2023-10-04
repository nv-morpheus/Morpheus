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


@pytest.fixture(scope="function", name="milvus_vdb_serivce_fixture")
def milvus_vdb_serivce():
    vdb_service = MilvusVectorDBService(uri="http://localhost:19530")
    return vdb_service


def create_milvus_collection(vdb_service: MilvusVectorDBService):
    collection_config = {
        "collection_conf": {
            "shards": 2,
            "auto_id": False,
            "consistency_level": "Strong",
            "description": "Test collection",
            "schema_conf": {
                "enable_dynamic_field": True,
                "schema_fields": [
                    {
                        "name": "id",
                        "dtype": "int64",
                        "description": "Primary key for the collection",
                        "is_primary": True,
                    },
                    {
                        "name": "embedding",
                        "dtype": "float_vector",
                        "description": "Embedding vectors",
                        "is_primary": False,
                        "dim": 10,
                    },
                    {
                        "name": "age",
                        "dtype": "int64",
                        "description ": "Age",
                        "is_primary": False,
                    },
                ],
                "description": "Test collection schema",
            },
        }
    }
    vdb_service.create(name="test", overwrite=True, **collection_config)


def create_milvus_collection_idx_part(vdb_service: MilvusVectorDBService):
    collection_config = {
        "collection_conf": {
            "shards": 2,
            "auto_id": False,
            "consistency_level": "Strong",
            "description": "Test collection with partition and index",
            "index_conf": {
                "field_name": "embedding", "metric_type": "L2"
            },
            "partition_conf": {
                "timeout": 1, "partitions": [{
                    "name": "age_partition", "description": "Partition by age"
                }]
            },
            "schema_conf": {
                "enable_dynamic_field": True,
                "schema_fields": [
                    {
                        "name": "id",
                        "dtype": "int64",
                        "description": "Primary key for the collection",
                        "is_primary": True,
                    },
                    {
                        "name": "embedding",
                        "dtype": "float_vector",
                        "description": "Embedding vectors",
                        "is_primary": False,
                        "dim": 10,
                    },
                    {
                        "name": "age",
                        "dtype": "int64",
                        "description ": "Age",
                        "is_primary": False,
                    },
                ],
                "description": "Test collection schema",
            },
        }
    }
    vdb_service.create(name="test_idx_part", overwrite=True, **collection_config)


@pytest.mark.use_cpp
def test_write_to_vector_db_stage_with_instance_pipe(milvus_vdb_serivce_fixture,
                                                     config: Config,
                                                     pipeline_batch_size: int = 256):
    config.pipeline_batch_size = pipeline_batch_size

    rows_count = 5
    dimensions = 10
    collection_name = "test"

    create_milvus_collection(milvus_vdb_serivce_fixture)

    df = cudf.DataFrame({
        "id": [i for i in range(rows_count)],
        "age": [random.randint(20, 40) for i in range(rows_count)],
        "embedding": [[random.random() for _ in range(dimensions)] for _ in range(rows_count)]
    })

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE, "module_name": "to_control_message", "namespace": MORPHEUS_MODULE_NAMESPACE
    }
    vdb_service = MilvusVectorDBService(uri="http://localhost:19530")

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        LinearModulesStage(config,
                           to_cm_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=ControlMessage))
    pipe.add_stage(WriteToVectorDBStage(config, resource_name=collection_name, service=vdb_service))

    pipe.run()

    actual_count = milvus_vdb_serivce_fixture.count(name=collection_name)
    milvus_vdb_serivce_fixture.close()

    assert actual_count == 5


@pytest.mark.use_cpp
def test_write_to_vector_db_stage_with_name_pipe(milvus_vdb_serivce_fixture,
                                                 config: Config,
                                                 pipeline_batch_size: int = 256):
    config.pipeline_batch_size = pipeline_batch_size

    create_milvus_collection_idx_part(milvus_vdb_serivce_fixture)

    rows_count = 5
    dimensions = 10
    collection_name = "test_idx_part"

    resource_kwargs = {"collection_conf": {"partition_name": "age_partition"}}

    df = cudf.DataFrame({
        "id": [i for i in range(rows_count)],
        "age": [random.randint(20, 40) for i in range(rows_count)],
        "embedding": [[random.random() for _ in range(dimensions)] for _ in range(rows_count)]
    })

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE, "module_name": "to_control_message", "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df], repeat=2))
    pipe.add_stage(
        LinearModulesStage(config,
                           to_cm_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=ControlMessage))
    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=collection_name,
                             service="milvus",
                             uri="http://localhost:19530",
                             resource_kwargs=resource_kwargs))

    pipe.run()

    actual_count = milvus_vdb_serivce_fixture.count(name=collection_name)
    milvus_vdb_serivce_fixture.close()

    assert actual_count == 10
