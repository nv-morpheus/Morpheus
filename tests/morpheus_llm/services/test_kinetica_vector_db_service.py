#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
import cudf

from morpheus_llm.service.vdb.kinetica_vector_db_service import KineticaVectorDBService



@pytest.mark.kinetica
def test_has_store_object(kinetica_service: KineticaVectorDBService):
    # Check if a non-existing collection exists in the Kinetica server.
    collection_name = "non_existing_collection"
    assert not kinetica_service.has_store_object(collection_name)


@pytest.mark.kinetica
def test_create_and_drop_collection(kinetica_type: list[list], kinetica_service: KineticaVectorDBService):
    collection_name = "test_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection and check if it exists.
    kinetica_service.create(collection_name, kinetica_type)
    assert kinetica_service.has_store_object(collection_name)

    # Drop the collection and check if it no longer exists.
    kinetica_service.drop(collection_name)
    assert not kinetica_service.has_store_object(collection_name)


@pytest.mark.kinetica
def test_insert_and_retrieve_by_keys(kinetica_service: KineticaVectorDBService,
                                     kinetica_type: list[list],
                                     kinetica_data: list[list]):
    collection_name = "test_insert_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data into the collection.
    response = kinetica_service.insert(collection_name, kinetica_data)
    assert response["insert_count"] == len(kinetica_data)

    # Retrieve inserted data by primary keys.
    keys_to_retrieve = [2, 4, 6]
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, keys_to_retrieve)
    assert len(retrieved_data) == len(keys_to_retrieve)

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_query(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = "test_search_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Define a search query.
    query = f"select * from {collection_name} where id=2 or id=6"

    # Perform a search in the collection.
    search_result = kinetica_service.query(collection_name, query)
    assert len(search_result.records) == 2

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
@pytest.mark.asyncio
async def test_similarity_search_with_data(kinetica_service: KineticaVectorDBService,
                                           kinetica_type: list[list],
                                           kinetica_data: list[list]):
    collection_name = "test_search_with_data_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

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

    assert len(search_result[0]) == 2
    assert sorted(list(search_result[0][0].keys())) == ["id", "metadata"]

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_count(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = "test_count_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

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

    collection_name = "test_overwrite_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data to the collection.
    response1 = kinetica_service.insert(collection_name, kinetica_data)
    assert response1["insert_count"] == len(kinetica_data)

    # Create the same collection again with overwrite=True.
    kinetica_service.create(collection_name, kinetica_type, overwrite=True)

    # Insert different data into the collection.
    data2 = [{"id": i, "embeddings": [i / 10] * 3, "age": 26 + i} for i in range(10)]

    response2 = kinetica_service.insert(collection_name, data2)
    assert response2["insert_count"] == len(data2)

    # Retrieve the data from the collection and check if it matches the second set of data.
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, list(range(10)))
    for i in range(10):
        assert retrieved_data[i]["age"] == data2[i]["age"]

    # Clean up the collection.
    kinetica_service.drop(collection_name)



@pytest.mark.kinetica
def test_update(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):
    collection_name = "test_update_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection with the specified schema configuration.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data to the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Use updated data to test the update/upsert functionality.
    updated_data = []

    # Apply update/upsert on updated_data.
    result_dict = kinetica_service.update(collection_name, updated_data)

    assert result_dict["count_updated"] == 7

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_delete_by_keys(kinetica_service: KineticaVectorDBService,
                        kinetica_type: list[list],
                        kinetica_data: list[list]):
    collection_name = "test_delete_by_keys_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Delete data by keys from the collection.
    keys_to_delete = [5, 6]
    kinetica_service.delete_by_keys(collection_name, keys_to_delete)

    response = kinetica_service.query(collection_name, query="id >= 0")
    assert len(response) == len(kinetica_data) - 2

    for item in response:
        assert item["id"] not in [5, 6]

    # Clean up the collection.
    kinetica_service.drop(collection_name)


@pytest.mark.kinetica
def test_delete(kinetica_service: KineticaVectorDBService, kinetica_type: list[list], kinetica_data: list[list]):

    collection_name = "test_delete_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)

    # Insert data into the collection.
    kinetica_service.insert(collection_name, kinetica_data)

    # Delete expression.
    delete_expr = "id in (0,1)"

    # Delete data from the collection using the expression.
    delete_response = kinetica_service.delete(collection_name, delete_expr)
    assert delete_response["delete_count"] == 2

    response = kinetica_service.query(collection_name, query="id > 0")
    assert len(response) == len(kinetica_data) - 2

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

    collection_name = "test_create_from_dataframe_collection"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection using dataframe schema.
    kinetica_service.create_from_dataframe(collection_name, df=df.to_pandas(), index_field="embedding")

    assert kinetica_service.has_store_object(collection_name)

    # Clean up the collection.
    kinetica_service.drop(collection_name)



@pytest.mark.kinetica
@pytest.mark.slow
def test_insert_dataframe(kinetica_service: KineticaVectorDBService,
                          kinetica_type: list[list], kinetica_data: list[list]):
    num_rows = len(kinetica_data)
    collection_name = "test_insert_dataframe"

    # Make sure to drop any existing collection from previous runs.
    kinetica_service.drop(collection_name)

    # Create a collection.
    kinetica_service.create(collection_name, kinetica_type)
    import pandas as pd
    df = pd.DataFrame(kinetica_data, columns=["id", "embeddings", "metadata"])

    kinetica_service.insert_dataframe(collection_name, df)

    # Retrieve inserted data by primary keys.
    retrieved_data = kinetica_service.retrieve_by_keys(collection_name, list(range(1, num_rows+1)))
    assert len(retrieved_data) == len(kinetica_data)

    # Clean up the collection.
    kinetica_service.drop(collection_name)
