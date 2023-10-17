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

import json
import os
import random

import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.modules import to_control_message  # noqa: F401 # pylint: disable=unused-import
from morpheus.pipeline import LinearPipeline
from morpheus.service.milvus_vector_db_service import MilvusVectorDBService
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.output.write_to_vector_db import WriteToVectorDBStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import TO_CONTROL_MESSAGE


@pytest.fixture(scope="function", name="milvus_service_fixture")
def milvus_service(milvus_server_uri: str):
    service = MilvusVectorDBService(uri=milvus_server_uri)
    yield service


def get_test_df(num_input_rows):

    df = cudf.DataFrame({
        "id": list(range(num_input_rows)),
        "age": [random.randint(20, 40) for i in range(num_input_rows)],
        "embedding": [[random.random() for _ in range(10)] for _ in range(num_input_rows)]
    })

    return df


def create_milvus_collection(collection_name: str, conf_file: str, service: MilvusVectorDBService):

    conf_filepath = os.path.join(TEST_DIRS.tests_data_dir, "service", conf_file)

    with open(conf_filepath, 'r', encoding="utf-8") as json_file:
        collection_config = json.load(json_file)

    service.create(name=collection_name, overwrite=True, **collection_config)


@pytest.mark.milvus
@pytest.mark.use_cpp
@pytest.mark.parametrize("use_instance, num_input_rows, expected_num_output_rows", [(True, 5, 5), (False, 5, 5)])
def test_write_to_vector_db_stage_pipe(milvus_service_fixture: MilvusVectorDBService,
                                       milvus_server_uri: str,
                                       use_instance: bool,
                                       config: Config,
                                       num_input_rows: int,
                                       expected_num_output_rows: int):

    collection_name = "test_stage_insert_collection"

    # Create milvus collection using config file.
    create_milvus_collection(collection_name, "milvus_idx_part_collection_conf.json", milvus_service_fixture)
    df = get_test_df(num_input_rows)

    to_cm_module_config = {
        "module_id": TO_CONTROL_MESSAGE, "module_name": "to_control_message", "namespace": MORPHEUS_MODULE_NAMESPACE
    }

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [df]))
    pipe.add_stage(
        LinearModulesStage(config,
                           to_cm_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=ControlMessage))

    # Provide partition name to insert data into the partition otherwise goes to '_default' partition.
    resource_kwargs = {"collection_conf": {"partition_name": "age_partition"}}

    if use_instance:
        # Instantiate stage with service instance and insert options.
        write_to_vdb_stage = WriteToVectorDBStage(config,
                                                  resource_name=collection_name,
                                                  service=milvus_service_fixture,
                                                  resource_kwargs=resource_kwargs)
    else:
        # Instantiate stage with service name, uri and insert options.
        write_to_vdb_stage = WriteToVectorDBStage(config,
                                                  resource_name=collection_name,
                                                  service="milvus",
                                                  uri=milvus_server_uri,
                                                  resource_kwargs=resource_kwargs)

    pipe.add_stage(write_to_vdb_stage)
    sink_stage = pipe.add_stage(InMemorySinkStage(config))
    pipe.run()

    messages = sink_stage.get_messages()

    assert len(messages) == 1
    assert isinstance(messages[0], ControlMessage)
    assert messages[0].has_metadata("insert_response")

    # Insert entities response as a dictionary.
    response = messages[0].get_metadata("insert_response")

    assert response["insert_count"] == expected_num_output_rows
    assert response["succ_count"] == expected_num_output_rows
    assert response["err_count"] == 0
