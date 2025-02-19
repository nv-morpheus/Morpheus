#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

import cudf

from morpheus_llm.service.vdb.kinetica_vector_db_service import KineticaVectorDBService

# These tests need a running instance of Kinetica
# Steps to run these tests
#    1. Create a Kinetica free SAAS account (https://cloud.kinetica.com/trynow/)
#    2. Set the `KINETICA_HOST` environment variable to the host address from the first step
#    3. Set the `KINETICA_USER` environment variable to your username
#    4. Set the `KINETICA_PASSWORD` environment variable to your password
#    5. Optionally set the `KINETICA_SCHEMA` to your schema name
#


@pytest.mark.kinetica
def test_has_store_object(kinetica_service: KineticaVectorDBService):
    # Check if a non-existing collection exists in the Kinetica server.
    collection_name = kinetica_service.collection_name("non_existing_collection")
    assert not kinetica_service.has_store_object(collection_name)


@pytest.mark.kinetica
def test_create_and_drop_collection(kinetica_type: list[list], kinetica_service: KineticaVectorDBService):
    collection_name = kinetica_service.collection_name("test_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection and check if it exists.
    kinetica_service.create(collection_name, table_type=kinetica_type)
    assert kinetica_service.has_store_object(collection_name)

    # Drop the collection and check if it no longer exists.
    kinetica_service.drop(collection_name)
    assert not kinetica_service.has_store_object(collection_name)


@pytest.mark.kinetica
def test_insert_and_retrieve_by_keys(kinetica_service: KineticaVectorDBService,
                                     kinetica_type: list[list],
                                     kinetica_data: list[list]):
    collection_name = kinetica_service.collection_name("test_insert_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data into the collection.
    response = kinetica_service.insert(collection_name, kinetica_data)
    assert response["count_inserted"] == len(kinetica_data)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, keys_to_retrieve)
    assert len(retrieved_data) == len(keys_to_retrieve)

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_query(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = kinetica_service.collection_name("test_search_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Define a search query.
    query = f"select * from {collection_name} where id=2 or id=6"

    # Perform a search in the collection.
    search_result = kinetica_service.query(collection_name, query)
    result_list = []
    for rec in search_result:
        result_list.append(rec)

    print(f"SEARCH RESULT = {result_list}")

    assert len(result_list) == 2

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
@pytest.mark.asyncio
async def test_similarity_search_with_data(kinetica_service: KineticaVectorDBService,
                                           kinetica_type: list[list],
                                           kinetica_data: list[list]):
    collection_name = kinetica_service.collection_name("test_search_with_data_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data to the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    rng = np.random.default_rng(seed=100)
    search_vec = rng.random((1, 3))

    # Define a search filter.
    expr = "id=2 or id=7"

    # Perform a search in the collection.
    similarity_search_coroutine = await kinetica_service.similarity_search(collection_name,
                                                                         embeddings=search_vec,
                                                                         expr=expr)
    search_result = await similarity_search_coroutine
    result_list = []
    for rec in search_result:
        result_list.append(rec)

    assert len(result_list) == 1

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_count(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = kinetica_service.collection_name("test_count_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Get the count of entities in the collection.
    count = kinetica_service.count(collection_name)
    assert count == len(kinetica_data)

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_overwrite_collection_on_create(kinetica_service: KineticaVectorDBService,
                                        kinetica_type: list[list],
                                        kinetica_data: list[list]):

    collection_name = kinetica_service.collection_name("test_overwrite_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data to the collection.
    response1 = kinetica_service.insert(collection_name, kinetica_data)
    assert response1["count_inserted"] == len(kinetica_data)

    # Create the same collection again with overwrite=True.
    kinetica_service.create(collection_name, overwrite=True, table_type=kinetica_type)

    # Insert different data into the collection.
    new_data = [[
        i+1,
        [random.random() for _ in range(3)],
        json.dumps({"metadata": f"New metadata for row {i+1}"}),
    ] for i in range(10)]

    response2 = kinetica_service.insert(collection_name, new_data)
    assert response2["count_inserted"] == len(new_data)

    # Retrieve the data from the collection and check if it matches the second set of data.
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, list(range(1, 11)))

    assert len(retrieved_data) == len(new_data)

    # Clean up the collection.
    kinetica_service.drop(collection_name)



@pytest.mark.kinetica
def test_update(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = kinetica_service.collection_name("test_update_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection with the specified schema configuration.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data to the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    expressions = ["id in (2, 4, 6)"]

    # Use updated data to test the update/upsert functionality.
    metadata = "New updated metadata for row id"
    updated_value = json.dumps({"metadata": f"{metadata}"})
    updated_data = {"embeddings": [random.random() for _ in range(3)], "metadata": f"{updated_value}"}

    # Apply update/upsert on updated_data.
    result_dict = kinetica_service.update(collection_name, [],
                                          expressions=expressions,
                                          new_values_maps=[updated_data],
                                          options={})

    assert result_dict["count_updated"] == 3

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_delete(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):

    collection_name = kinetica_service.collection_name("test_delete_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Delete expression.
    delete_expr = "id in (0,1)"

    # Delete data from the collection using the expression.
    delete_response = kinetica_service.delete(collection_name, delete_expr)
    assert delete_response["count_deleted"] == 1

    response = kinetica_service.query(collection_name, f"select * from {collection_name} where id > 0")
    result_list = []
    for rec in response:
        result_list.append(rec)

    assert len(result_list) == len(kinetica_data) - 1

    for item in response:
        assert item["id"] > 1

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_create_from_dataframe(kinetica_service: KineticaVectorDBService):

    df = cudf.DataFrame({
        "id": list(range(10)),
        "age": [random.randint(20, 40) for i in range(10)],
        "embedding": [[random.random() for _ in range(10)] for _ in range(10)]
    })

    collection_name = kinetica_service.collection_name("test_create_from_dataframe_collection")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection using dataframe schema.
    kinetica_service.create_from_dataframe(collection_name, df=df.to_pandas(), index_field="embedding")

    assert kinetica_service.has_store_object(collection_name)

    # Clean up the collection.
    kinetica_service.drop(collection_name)



@pytest.mark.kinetica
def test_insert_dataframe(kinetica_service: KineticaVectorDBService,
                          kinetica_type: list[list], kinetica_data: list[list]):
    num_rows = len(kinetica_data)
    collection_name = kinetica_service.collection_name("test_insert_dataframe")

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, table_type=kinetica_type)
    import pandas as pd
    df = pd.DataFrame(kinetica_data, columns=["id", "embeddings", "metadata"])

    kinetica_service.insert_dataframe(collection_name, df)

    # Retrieve inserted data by primary keys.
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, list(range(1, num_rows+1)))
    assert len(retrieved_data) == len(kinetica_data)

    # Clean up the collection.
    kinetica_service.drop(collection_name)
