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

import json
import random

import numpy as np
import pymilvus
import pytest

import cudf

from morpheus.service.vdb.milvus_client import MILVUS_DATA_TYPE_MAP
from morpheus.service.vdb.milvus_vector_db_service import FieldSchemaEncoder
from morpheus.service.vdb.milvus_vector_db_service import MilvusVectorDBService


@pytest.fixture(scope="module", name="milvus_service")
def milvus_service_fixture(milvus_server_uri: str):
    # This fixture is scoped to the function level since the WriteToVectorDBStage will close the connection on'
    # pipeline completion
    service = MilvusVectorDBService(uri=milvus_server_uri)
    yield service


@pytest.mark.milvus
def test_list_store_objects(milvus_service: MilvusVectorDBService):
    # List all collections in the Milvus server.
    collections = milvus_service.list_store_objects()
    assert isinstance(collections, list)


@pytest.mark.milvus
def test_has_store_object(milvus_service: MilvusVectorDBService):
    # Check if a non-existing collection exists in the Milvus server.
    collection_name = "non_existing_collection"
    assert not milvus_service.has_store_object(collection_name)


@pytest.fixture(scope="module", name="sample_field")
def sample_field_fixture():
    return pymilvus.FieldSchema(name="test_field", dtype=pymilvus.DataType.INT64)


@pytest.mark.milvus
def test_create_and_drop_collection(idx_part_collection_config: dict, milvus_service: MilvusVectorDBService):
    collection_name = "test_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection and check if it exists.
    milvus_service.create(collection_name, **idx_part_collection_config)
    assert milvus_service.has_store_object(collection_name)

    # Drop the collection and check if it no longer exists.
    milvus_service.drop(collection_name)
    assert not milvus_service.has_store_object(collection_name)


@pytest.mark.milvus
def test_insert_and_retrieve_by_keys(milvus_service: MilvusVectorDBService,
                                     idx_part_collection_config: dict,
                                     milvus_data: list[dict]):
    collection_name = "test_insert_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    response = milvus_service.insert(collection_name, milvus_data)
    assert response["insert_count"] == len(milvus_data)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = milvus_service.retrieve_by_keys(collection_name, keys_to_retrieve)
    assert len(retrieved_data) == len(keys_to_retrieve)

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_query(milvus_service: MilvusVectorDBService, idx_part_collection_config: dict, milvus_data: list[dict]):
    collection_name = "test_search_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Define a search query.
    query = "age==26 or age==27"

    # Perform a search in the collection.
    search_result = milvus_service.query(collection_name, query)
    assert len(search_result) == 2
    assert search_result[0]["age"] in [26, 27]
    assert search_result[1]["age"] in [26, 27]

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
@pytest.mark.asyncio
async def test_similarity_search_with_data(milvus_service: MilvusVectorDBService,
                                           idx_part_collection_config: dict,
                                           milvus_data: list[dict]):
    collection_name = "test_search_with_data_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data to the collection.
    milvus_service.insert(collection_name, milvus_data)

    rng = np.random.default_rng(seed=100)
    search_vec = rng.random((1, 3))

    # Define a search filter.
    expr = "age==26 or age==27"

    # Perform a search in the collection.
    similarity_search_coroutine = await milvus_service.similarity_search(collection_name,
                                                                         embeddings=search_vec,
                                                                         expr=expr)
    search_result = await similarity_search_coroutine

    assert len(search_result[0]) == 2
    assert search_result[0][0]["age"] in [26, 27]
    assert search_result[0][1]["age"] in [26, 27]
    assert len(search_result[0][0].keys()) == 2
    assert sorted(list(search_result[0][0].keys())) == ["age", "id"]

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_count(milvus_service: MilvusVectorDBService, idx_part_collection_config: dict, milvus_data: list[dict]):
    collection_name = "test_count_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Get the count of entities in the collection.
    count = milvus_service.count(collection_name)
    assert count == len(milvus_data)

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_overwrite_collection_on_create(milvus_service: MilvusVectorDBService,
                                        idx_part_collection_config: dict,
                                        milvus_data: list[dict]):

    collection_name = "test_overwrite_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data to the collection.
    response1 = milvus_service.insert(collection_name, milvus_data)
    assert response1["insert_count"] == len(milvus_data)

    # Create the same collection again with overwrite=True.
    milvus_service.create(collection_name, overwrite=True, **idx_part_collection_config)

    # Insert different data into the collection.
    data2 = [{"id": i, "embedding": [i / 10] * 3, "age": 26 + i} for i in range(10)]

    response2 = milvus_service.insert(collection_name, data2)
    assert response2["insert_count"] == len(data2)

    # Retrieve the data from the collection and check if it matches the second set of data.
    retrieved_data = milvus_service.retrieve_by_keys(collection_name, list(range(10)))
    for i in range(10):
        assert retrieved_data[i]["age"] == data2[i]["age"]

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_insert_into_partition(milvus_service: MilvusVectorDBService,
                               idx_part_collection_config: dict,
                               milvus_data: list[dict]):
    collection_name = "test_partition_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    partition_name = idx_part_collection_config["partition_conf"]["partitions"][0]["name"]

    # Create a collection with a partition.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the specified partition.
    response = milvus_service.insert(collection_name, milvus_data, partition_name=partition_name)
    assert response["insert_count"] == len(milvus_data)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = milvus_service.retrieve_by_keys(collection_name,
                                                     keys_to_retrieve,
                                                     partition_names=[partition_name])
    assert len(retrieved_data) == len(keys_to_retrieve)

    retrieved_data_default_part = milvus_service.retrieve_by_keys(collection_name,
                                                                  keys_to_retrieve,
                                                                  partition_names=["_default"])
    assert len(retrieved_data_default_part) == 0
    assert len(retrieved_data_default_part) != len(keys_to_retrieve)

    # Raises error if resource is partition and not passed partition name.
    with pytest.raises(ValueError, match="Mandatory argument 'partition_name' is required when resource='partition'"):
        milvus_service.drop(name=collection_name, resource="partition")

    # Clean up the partition
    milvus_service.drop(name=collection_name, resource="partition", partition_name=partition_name)

    # Raises error if resource is index and not passed partition name.
    with pytest.raises(ValueError,
                       match="Mandatory arguments 'field_name' and 'index_name' are required when resource='index'"):
        milvus_service.drop(name=collection_name, resource="index")

    milvus_service.drop(name=collection_name, resource="index", field_name="embedding", index_name="_default_idx_")

    retrieved_data_after_part_drop = milvus_service.retrieve_by_keys(collection_name, keys_to_retrieve)
    assert len(retrieved_data_after_part_drop) == 0

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_update(milvus_service: MilvusVectorDBService, simple_collection_config: dict, milvus_data: list[dict]):
    collection_name = "test_update_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection with the specified schema configuration.
    milvus_service.create(collection_name, **simple_collection_config)

    # Insert data to the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Use updated data to test the update/upsert functionality.
    updated_data = [{
        "type": MILVUS_DATA_TYPE_MAP["int64"], "name": "id", "values": list(range(5, 12))
    },
                    {
                        "type": MILVUS_DATA_TYPE_MAP["float_vector"],
                        "name": "embedding",
                        "values": [[i / 5.0] * 3 for i in range(5, 12)]
                    }, {
                        "type": MILVUS_DATA_TYPE_MAP["int64"], "name": "age", "values": [25 + i for i in range(5, 12)]
                    }]

    # Apply update/upsert on updated_data.
    result_dict = milvus_service.update(collection_name, updated_data)

    assert result_dict["upsert_count"] == 7
    assert result_dict["insert_count"] == 7
    assert result_dict["succ_count"] == 7

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_delete_by_keys(milvus_service: MilvusVectorDBService,
                        idx_part_collection_config: dict,
                        milvus_data: list[dict]):
    collection_name = "test_delete_by_keys_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Delete data by keys from the collection.
    keys_to_delete = [5, 6]
    milvus_service.delete_by_keys(collection_name, keys_to_delete)

    response = milvus_service.query(collection_name, query="id >= 0")
    assert len(response) == len(milvus_data) - 2

    for item in response:
        assert item["id"] not in [5, 6]

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_delete(milvus_service: MilvusVectorDBService, idx_part_collection_config: dict, milvus_data: list[dict]):

    collection_name = "test_delete_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Delete expression.
    delete_expr = "id in [0,1]"

    # Delete data from the collection using the expression.
    delete_response = milvus_service.delete(collection_name, delete_expr)
    assert delete_response["delete_count"] == 2

    response = milvus_service.query(collection_name, query="id > 0")
    assert len(response) == len(milvus_data) - 2

    for item in response:
        assert item["id"] > 1

    # Clean up the collection.
    milvus_service.drop(collection_name)


@pytest.mark.milvus
def test_release_collection(milvus_service: MilvusVectorDBService,
                            idx_part_collection_config: dict,
                            milvus_data: list[dict]):
    collection_name = "test_release_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection.
    milvus_service.create(collection_name, **idx_part_collection_config)

    # Insert data into the collection.
    milvus_service.insert(collection_name, milvus_data)

    # Release resource from the memory.
    milvus_service.release_resource(name=collection_name)


def test_get_collection_lock():
    """
    This test doesn't require milvus server to be running.
    """
    collection_name = "test_collection_lock"
    lock = MilvusVectorDBService.get_collection_lock(collection_name)
    assert "lock" == type(lock).__name__
    assert collection_name in MilvusVectorDBService._collection_locks


@pytest.mark.milvus
def test_create_from_dataframe(milvus_service: MilvusVectorDBService):

    df = cudf.DataFrame({
        "id": list(range(10)),
        "age": [random.randint(20, 40) for i in range(10)],
        "embedding": [[random.random() for _ in range(10)] for _ in range(10)]
    })

    collection_name = "test_create_from_dataframe_collection"

    # Make sure to drop any existing collection from previous runs.
    milvus_service.drop(collection_name)

    # Create a collection using dataframe schema.
    milvus_service.create_from_dataframe(collection_name, df=df, index_field="embedding")

    assert milvus_service.has_store_object(collection_name)

    # Clean up the collection.
    milvus_service.drop(collection_name)


def test_fse_default():
    encoder = FieldSchemaEncoder()
    result = encoder.default(pymilvus.DataType.INT32)
    assert result == "DataType.INT32"


def test_fse_object_hook():
    data = {"name": "test_field", "type": "DataType.INT64"}
    result = FieldSchemaEncoder.object_hook(data)
    assert result["type"] == pymilvus.DataType.INT64


def test_fse_load(tmp_path):
    data = {"name": "test_field", "type": "DataType.INT64"}
    file_path = tmp_path / "test.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with open(file_path, "r", encoding="utf-8") as f:
        result = FieldSchemaEncoder.load(f)
    assert result.name == "test_field"
    assert result.dtype == pymilvus.DataType.INT64


def test_fse_loads():
    data = '{"name": "test_field", "type": "DataType.INT64"}'
    result = FieldSchemaEncoder.loads(data)
    assert result.name == "test_field"
    assert result.dtype == pymilvus.DataType.INT64


def test_fse_from_dict():
    data = {"name": "test_field", "dtype": "DataType.INT64"}
    result = FieldSchemaEncoder.from_dict(data)
    assert result.name == "test_field"
    assert result.dtype == pymilvus.DataType.INT64
