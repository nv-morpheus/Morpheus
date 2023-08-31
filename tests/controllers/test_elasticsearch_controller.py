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

import pickle
import time
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from elasticsearch import Elasticsearch

from morpheus.controllers.elasticsearch_controller import ElasticsearchController

# pylint: disable=W0621


# Define a mock function for _apply_derive_params_function
def mock_derive_params(kwargs):
    kwargs["retry_on_status"] = 3
    kwargs["retry_on_timeout"] = 3 * 10

    return kwargs


@pytest.fixture(scope="module")
def mock_es_controller(connection_kwargs):
    with patch("morpheus.controllers.elasticsearch_controller.Elasticsearch", Mock(spec=Elasticsearch)):
        controller = ElasticsearchController("test_index", connection_kwargs, refresh_period_secs=10)
        yield controller


@pytest.fixture(scope="module")
def connection_kwargs():
    kwargs = {"hosts": [{"host": "localhost", "port": 9200, "scheme": "http"}]}
    yield kwargs


@pytest.mark.use_python
def test_constructor(mock_es_controller, connection_kwargs):
    assert mock_es_controller._index == "test_index"
    assert mock_es_controller._connection_kwargs == connection_kwargs
    assert mock_es_controller._raise_on_exception is False
    assert mock_es_controller._refresh_period_secs == 10
    assert mock_es_controller._client is not None


@pytest.mark.use_python
def test_apply_derive_params_func(mock_es_controller):
    pickled_func_str = str(pickle.dumps(mock_derive_params), encoding="latin1")

    # Create pickled_func_config
    pickled_func_config = {
        "pickled_func_str": pickled_func_str,  # Pickled function string
        "encoding": "latin1"
    }

    # Apply the mock function and check if connection_kwargs is updated
    mock_es_controller._apply_derive_params_func(pickled_func_config)
    assert mock_es_controller._connection_kwargs == {
        "hosts": [{
            "host": "localhost", "port": 9200, "scheme": "http"
        }], "retry_on_status": 3, "retry_on_timeout": 30
    }


@pytest.mark.use_python
def test_refresh_client_force(mock_es_controller):
    # Simulate a force refresh
    mock_es_controller.refresh_client(force=True)

    assert mock_es_controller._client is not None
    assert mock_es_controller._last_refresh_time > 0
    assert isinstance(mock_es_controller._client, Mock)


@pytest.mark.use_python
def test_refresh_client_not_needed(mock_es_controller):
    # Set last refresh time to a recent time
    current_time = time.time()
    mock_es_controller._last_refresh_time = current_time

    # Simulate a refresh not needed scenario
    mock_es_controller.refresh_client()
    # Assert client is not None
    assert mock_es_controller._client is not None
    # Assert last_refresh_time is unchanged
    assert mock_es_controller._last_refresh_time == current_time
    # Assert client type remains the same
    assert isinstance(mock_es_controller._client, Mock)


@pytest.mark.use_python
def test_refresh_client_needed(mock_es_controller):
    # Set last refresh time to a recent time
    current_time = time.time()
    mock_es_controller._refresh_period_secs = 1
    mock_es_controller._last_refresh_time = current_time
    time.sleep(1)

    # Simulate a refresh needed scenario
    mock_es_controller.refresh_client()

    # Assert client is not None
    assert mock_es_controller._client is not None
    # Assert last_refresh_time is changed
    assert mock_es_controller._last_refresh_time > current_time
    # Assert client type remains the same
    assert isinstance(mock_es_controller._client, Mock)


@pytest.mark.use_python
@patch("morpheus.controllers.elasticsearch_controller.parallel_bulk", return_value=[(True, None)])
def test_parallel_bulk_write(mock_parallel_bulk, mock_es_controller):
    # Define your mock actions
    mock_actions = [{"_index": "test_index", "_id": 1, "_source": {"field1": "value1"}}]

    mock_es_controller.parallel_bulk_write(actions=mock_actions)
    mock_parallel_bulk.assert_called_once()


def test_search_documents_success(mock_es_controller):
    mock_es_controller._client.search.return_value = {"hits": {"total": 1, "hits": [{"_source": {"field1": "value1"}}]}}
    query = {"match": {"field1": "value1"}}
    result = mock_es_controller.search_documents(query=query)
    assert isinstance(result, dict)
    assert "hits" in result
    assert "total" in result["hits"]
    assert result["hits"]["total"] == 1


def test_search_documents_failure_supress_errors(mock_es_controller):
    mock_es_controller._client.search.side_effect = ConnectionError("Connection error")
    query = {"match": {"field1": "value1"}}
    result = mock_es_controller.search_documents(query=query)
    assert isinstance(result, dict)
    assert not result


def test_search_documents_failure_raise_error(mock_es_controller):
    mock_es_controller._raise_on_exception = True
    query = {"match": {"field1": "value1"}}
    with pytest.raises(RuntimeError):
        mock_es_controller.search_documents(query=query)
