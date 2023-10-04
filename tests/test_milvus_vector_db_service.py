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
import typing
from os import path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.service.milvus_client import MilvusClient
from morpheus.service.milvus_vector_db_service import MilvusVectorDBService


@pytest.fixture(scope="function", name="mock_milvus_client_fixture")
def mock_milvus_client() -> MilvusClient:
    with patch('morpheus.service.milvus_vector_db_service.MilvusClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture(scope="function", name="milvus_service_fixture")
def milvus_service(mock_milvus_client_fixture) -> MilvusVectorDBService:
    service = MilvusVectorDBService(uri="http://localhost:19530")
    return service


@pytest.mark.parametrize(
    "has_store_object_input, expected_result",
    [
        (True, True),  # Collection exists
        (False, False),  # Collection does not exist
    ])
def test_has_store_object(milvus_service_fixture: MilvusVectorDBService,
                          mock_milvus_client_fixture: MilvusClient,
                          has_store_object_input: bool,
                          expected_result: bool):
    mock_milvus_client_fixture.has_collection.return_value = has_store_object_input
    assert milvus_service_fixture.has_store_object("test_collection") == expected_result


@pytest.mark.parametrize(
    "list_collections_input, expected_result",
    [
        (["collection1", "collection2"], ["collection1", "collection2"]),  # Collections exist
        ([], []),  # No collections exist
    ])
def test_list_store_objects(milvus_service_fixture: MilvusVectorDBService,
                            mock_milvus_client_fixture: MilvusClient,
                            list_collections_input: list,
                            expected_result: list):
    mock_milvus_client_fixture.list_collections.return_value = list_collections_input
    assert milvus_service_fixture.list_store_objects() == expected_result


@pytest.mark.parametrize("overwrite, has_collection", [(True, True), (False, False), (True, False)])
def test_create(milvus_service_fixture: MilvusVectorDBService,
                mock_milvus_client_fixture: MilvusClient,
                overwrite: bool,
                has_collection: bool):
    filepath = path.join(TEST_DIRS.tests_data_dir, "service", "milvus_test_collection_conf.json")
    collection_config = {}
    with open(filepath, "r") as file:
        collection_config = json.load(file)

    mock_milvus_client_fixture.has_collection.return_value = has_collection
    name = collection_config.pop("name")
    milvus_service_fixture.create(name=name, overwrite=overwrite, **collection_config)

    if overwrite:
        if has_collection:
            mock_milvus_client_fixture.drop_collection.assert_called_once()
        else:
            mock_milvus_client_fixture.drop_collection.assert_not_called()
    else:
        mock_milvus_client_fixture.drop_collection.assert_not_called()

    mock_milvus_client_fixture.create_collection_with_schema.assert_called_once()


def test_insert(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    data = [
        {
            "id": 1, "embedding": [0.1, 0.2, 0.3], "age": 30
        },
        {
            "id": 2, "embedding": [0.4, 0.5, 0.6], "age": 25
        },
    ]
    milvus_service_fixture.insert(name="test_collection", data=data)
    mock_milvus_client_fixture.get_collection.assert_called_once()


@pytest.mark.parametrize(
    "insert_data, expected_exception",
    [
        ([], RuntimeError),  # Collection does not exist
        ([], RuntimeError),  # Other error scenario
    ])
def test_insert_error(milvus_service_fixture: MilvusVectorDBService,
                      mock_milvus_client_fixture: MilvusClient,
                      insert_data: list,
                      expected_exception: Exception):
    mock_milvus_client_fixture.has_collection.return_value = False
    with pytest.raises(expected_exception):
        milvus_service_fixture.insert("non_existent_collection", data=insert_data)


def test_insert_dataframe(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    mock_insert = Mock()
    mock_milvus_client_fixture.get_collection.return_value.insert = mock_insert

    data = cudf.DataFrame({
        "id": [1, 2],
        "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "age": [30, 25],
    })

    milvus_service_fixture.insert_dataframe(name="test_collection", df=data)
    mock_insert.assert_called_once()


search_test_cases = [
    ("query", {
        "bool": {
            "must": [{
                "vector": {
                    "embedding": [0.1, 0.2, 0.3]
                }
            }]
        }
    }, "query_result"),
    ("data", [{
        "id": 1, "embedding": [0.1, 0.2, 0.3]
    }, {
        "id": 2, "embedding": [0.4, 0.5, 0.6]
    }], "data_result"),
    ("error_value", None, None),
    ("error_value", None, []),
]


@pytest.mark.parametrize("test_type, query_or_data, expected_result", search_test_cases)
def test_search(milvus_service_fixture: MilvusVectorDBService,
                mock_milvus_client_fixture: MilvusClient,
                test_type: str,
                query_or_data: dict,
                expected_result: typing.Any):
    if test_type == "query":
        mock_milvus_client_fixture.query.return_value = {"result": expected_result}
    elif test_type == "data":
        mock_milvus_client_fixture.search.return_value = {"result": expected_result}

    if test_type == "error_value":
        with pytest.raises(RuntimeError):
            milvus_service_fixture.search(name="test_collection", query=query_or_data, data=None)
    else:
        result = milvus_service_fixture.search(name="test_collection", **{test_type: query_or_data})
        assert result == {"result": expected_result}
        mock_milvus_client_fixture.load_collection.assert_called_once()
        if test_type == "query":
            mock_milvus_client_fixture.query.assert_called_once()
        elif test_type == "data":
            mock_milvus_client_fixture.search.assert_called_once()
        mock_milvus_client_fixture.release_collection.assert_called_once()


def test_update(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    data = [
        {
            "id": 1, "embedding": [0.1, 0.2, 0.3], "age": 30
        },
        {
            "id": 2, "embedding": [0.4, 0.5, 0.6], "age": 25
        },
    ]
    milvus_service_fixture.update(name="test_collection", data=data)
    mock_milvus_client_fixture.upsert.assert_called_once()


def test_delete_by_keys(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    keys = [1, 2]
    mock_milvus_client_fixture.delete.return_value = keys

    result = milvus_service_fixture.delete_by_keys(name="test_collection", keys=keys)
    assert result == keys
    mock_milvus_client_fixture.delete.assert_called_once()


def test_delete(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    expr = "age < 30"
    milvus_service_fixture.delete(name="test_collection", expr=expr)
    mock_milvus_client_fixture.delete_by_expr.assert_called_once()


def test_retrieve_by_keys(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    keys = [1]
    mock_milvus_client_fixture.get.return_value = [{"id": 1, "embedding": [0.1, 0.2, 0.3], "age": 30}]

    result = milvus_service_fixture.retrieve_by_keys(name="test_collection", keys=keys)
    assert result == [{"id": 1, "embedding": [0.1, 0.2, 0.3], "age": 30}]
    mock_milvus_client_fixture.get.assert_called_once()


def test_count(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    mock_milvus_client_fixture.num_entities.return_value = 5
    count = milvus_service_fixture.count(name="test_collection")
    assert count == 5
    mock_milvus_client_fixture.num_entities.assert_called_once()


def test_drop(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    milvus_service_fixture.drop(name="test_collection")
    mock_milvus_client_fixture.drop_collection.assert_called_once()


def test_describe(milvus_service_fixture: MilvusVectorDBService, mock_milvus_client_fixture: MilvusClient):
    mock_milvus_client_fixture.describe_collection.return_value = {"name": "test_collection"}
    description = milvus_service_fixture.describe(name="test_collection")
    assert description == {"name": "test_collection"}
    mock_milvus_client_fixture.describe_collection.assert_called_once()
