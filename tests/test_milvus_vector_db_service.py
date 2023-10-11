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

import concurrent.futures
import json
import os

import numpy as np
import pytest

from _utils import TEST_DIRS
from morpheus.service.milvus_client import MILVUS_DATA_TYPE_MAP
from morpheus.service.milvus_vector_db_service import MilvusVectorDBService


@pytest.fixture(scope="module", name="milvus_service_fixture")
def milvus_service(milvus_server_uri: str):
    service = MilvusVectorDBService(uri=milvus_server_uri)
    yield service


def load_yaml(filename):
    conf_filepath = os.path.join(TEST_DIRS.tests_data_dir, "service", filename)

    with open(conf_filepath, 'r', encoding="utf-8") as json_file:
        collection_config = json.load(json_file)

        return collection_config


@pytest.fixture(scope="module", name="data_fixture")
def data():
    inital_data = [{"id": i, "embedding": [i / 10.0] * 10, "age": 25 + i} for i in range(10)]
    yield inital_data


@pytest.fixture(scope="module", name="idx_part_collection_config_fixture")
def idx_part_collection_config():
    collection_config = load_yaml(filename="milvus_idx_part_collection_conf.json")
    yield collection_config


@pytest.fixture(scope="module", name="simple_collection_config_fixture")
def simple_collection_config():
    collection_config = load_yaml(filename="milvus_simple_collection_conf.json")
    yield collection_config


@pytest.mark.slow
def test_list_store_objects(milvus_service_fixture: MilvusVectorDBService):
    # List all collections in the Milvus server.
    collections = milvus_service_fixture.list_store_objects()
    assert isinstance(collections, list)


@pytest.mark.slow
def test_has_store_object(milvus_service_fixture: MilvusVectorDBService):
    # Check if a non-existing collection exists in the Milvus server.
    collection_name = "non_existing_collection"
    assert not milvus_service_fixture.has_store_object(collection_name)


@pytest.mark.slow
def test_create_and_drop_collection(milvus_service_fixture: MilvusVectorDBService,
                                    idx_part_collection_config_fixture: dict):
    # Create a collection and check if it exists.
    collection_name = "test_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)
    assert milvus_service_fixture.has_store_object(collection_name)

    # Drop the collection and check if it no longer exists.
    milvus_service_fixture.drop(collection_name)
    assert not milvus_service_fixture.has_store_object(collection_name)


@pytest.mark.slow
def test_insert_and_retrieve_by_keys(milvus_service_fixture: MilvusVectorDBService,
                                     idx_part_collection_config_fixture: dict,
                                     data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_insert_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the collection.
    response = milvus_service_fixture.insert(collection_name, data_fixture)
    assert response["insert_count"] == len(data_fixture)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = milvus_service_fixture.retrieve_by_keys(collection_name, keys_to_retrieve)
    assert len(retrieved_data) == len(keys_to_retrieve)

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_search(milvus_service_fixture: MilvusVectorDBService,
                idx_part_collection_config_fixture: dict,
                data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_search_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    # Define a search query.
    query = "age==26 or age==27"

    # Perform a search in the collection.
    search_result = milvus_service_fixture.search(collection_name, query)
    assert len(search_result) == 2
    assert search_result[0]["age"] in [26, 27]
    assert search_result[1]["age"] in [26, 27]

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_search_with_data(milvus_service_fixture: MilvusVectorDBService,
                          idx_part_collection_config_fixture: dict,
                          data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_search_with_data_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data to the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    rng = np.random.default_rng(seed=100)
    search_vec = rng.random((1, 10))

    # Define a search filter.
    fltr = "age==26 or age==27"

    # Perform a search in the collection.
    search_result = milvus_service_fixture.search(collection_name,
                                                  data=search_vec,
                                                  filter=fltr,
                                                  output_fields=["id", "age"])

    assert len(search_result[0]) == 2
    assert search_result[0][0]["entity"]["age"] in [26, 27]
    assert search_result[0][1]["entity"]["age"] in [26, 27]
    assert len(search_result[0][0]["entity"].keys()) == 2
    assert sorted(list(search_result[0][0]["entity"].keys())) == ["age", "id"]

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_count(milvus_service_fixture: MilvusVectorDBService,
               idx_part_collection_config_fixture: dict,
               data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_count_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    # Get the count of entities in the collection.
    count = milvus_service_fixture.count(collection_name)
    assert count == len(data_fixture)

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_overwrite_collection_on_create(milvus_service_fixture: MilvusVectorDBService,
                                        idx_part_collection_config_fixture: dict,
                                        data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_overwrite_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data to the collection.
    response1 = milvus_service_fixture.insert(collection_name, data_fixture)
    assert response1["insert_count"] == len(data_fixture)

    # Create the same collection again with overwrite=True.
    milvus_service_fixture.create(collection_name, overwrite=True, **idx_part_collection_config_fixture)

    # Insert different data into the collection.
    data2 = [{"id": i, "embedding": [i / 10] * 10, "age": 26 + i} for i in range(10)]

    response2 = milvus_service_fixture.insert(collection_name, data2)
    assert response2["insert_count"] == len(data2)

    # Retrieve the data from the collection and check if it matches the second set of data.
    retrieved_data = milvus_service_fixture.retrieve_by_keys(collection_name, list(range(10)))
    for i in range(10):
        assert retrieved_data[i]["age"] == data2[i]["age"]

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_insert_into_partition(milvus_service_fixture: MilvusVectorDBService,
                               idx_part_collection_config_fixture: dict,
                               data_fixture: list[dict]):
    # Create a collection with a partition.
    collection_name = "test_partition_collection"
    partition_name = idx_part_collection_config_fixture["collection_conf"]["partition_conf"]["partitions"][0]["name"]
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the specified partition.
    response = milvus_service_fixture.insert(collection_name,
                                             data_fixture,
                                             collection_conf={"partition_name": partition_name})
    assert response["insert_count"] == len(data_fixture)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = milvus_service_fixture.retrieve_by_keys(collection_name,
                                                             keys_to_retrieve,
                                                             partition_names=[partition_name])
    assert len(retrieved_data) == len(keys_to_retrieve)

    retrieved_data_default_part = milvus_service_fixture.retrieve_by_keys(collection_name,
                                                                          keys_to_retrieve,
                                                                          partition_names=["_default"])
    assert len(retrieved_data_default_part) == 0
    assert len(retrieved_data_default_part) != len(keys_to_retrieve)

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_update(milvus_service_fixture: MilvusVectorDBService,
                simple_collection_config_fixture: dict,
                data_fixture: list[dict]):
    collection_name = "test_update_collection"

    # Create a collection with the specified schema configuration.
    milvus_service_fixture.create(collection_name, **simple_collection_config_fixture)

    # Insert data to the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    # Use updated data to test the update/upsert functionality.
    updated_data = [{
        "type": MILVUS_DATA_TYPE_MAP["int64"], "name": "id", "values": list(range(5, 12))
    },
                    {
                        "type": MILVUS_DATA_TYPE_MAP["float_vector"],
                        "name": "embedding",
                        "values": [[i / 5.0] * 10 for i in range(5, 12)]
                    }, {
                        "type": MILVUS_DATA_TYPE_MAP["int64"], "name": "age", "values": [25 + i for i in range(5, 12)]
                    }]

    # Apply update/upsert on updated_data.
    result_dict = milvus_service_fixture.update(collection_name, updated_data)

    assert result_dict["upsert_count"] == 7
    assert result_dict["insert_count"] == 7
    assert result_dict["succ_count"] == 7

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_delete_by_keys(milvus_service_fixture: MilvusVectorDBService,
                        idx_part_collection_config_fixture: dict,
                        data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_delete_by_keys_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    # Delete data by keys from the collection.
    keys_to_delete = [2, 4, 6]
    response = milvus_service_fixture.delete_by_keys(collection_name, keys_to_delete)
    assert response == keys_to_delete

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_delete(milvus_service_fixture: MilvusVectorDBService,
                idx_part_collection_config_fixture: dict,
                data_fixture: list[dict]):
    # Create a collection.
    collection_name = "test_delete_collection"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    # Insert data into the collection.
    milvus_service_fixture.insert(collection_name, data_fixture)

    # Delete expression.
    delete_expr = "id in [0,1]"

    # Delete data from the collection using the expression.
    milvus_service_fixture.delete(collection_name, delete_expr)

    response = milvus_service_fixture.search(collection_name, query="id > 0")

    assert len(response) == len(data_fixture) - 2

    for item in response:
        assert item["id"] > 1

    # Clean up the collection.
    milvus_service_fixture.drop(collection_name)


@pytest.mark.slow
def test_with_collection_lock(milvus_service_fixture: MilvusVectorDBService,
                              idx_part_collection_config_fixture: dict,
                              data_fixture: list[dict]):

    # Create a collection.
    collection_name = "test_insert_and_search_order_with_collection_lock"
    milvus_service_fixture.create(collection_name, **idx_part_collection_config_fixture)

    execution_order = []

    def insert_data():
        result = milvus_service_fixture.insert(collection_name, data_fixture)
        assert result['insert_count'] == len(data_fixture)
        execution_order.append("Insert Executed")

    def search_data():
        filter_query = "age == 26 or age == 27"
        search_vec = np.random.random((1, 10))
        result = milvus_service_fixture.search(collection_name, data=search_vec, filter=filter_query)
        execution_order.append("Search Executed")
        assert isinstance(result, list)

    def count_entities():
        milvus_service_fixture.count(collection_name)
        execution_order.append("Count Collection Entities Executed")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(insert_data)
        executor.submit(search_data)
        executor.submit(count_entities)

    # Assert the execution order
    assert execution_order == ["Count Collection Entities Executed", "Insert Executed", "Search Executed"]
