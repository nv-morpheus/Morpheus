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
import typing
from unittest.mock import patch

import pytest

from morpheus.controllers.elasticsearch_controller import ElasticsearchController


# Define a mock function for _apply_derive_params_function
def mock_derive_params(kwargs):
    kwargs["retry_on_status"] = 3
    kwargs["retry_on_timeout"] = 3 * 10

    return kwargs


@pytest.fixture(scope="function", autouse=True)
def patch_elasticsearch():
    with patch("morpheus.controllers.elasticsearch_controller.Elasticsearch", autospec=True):
        yield


@pytest.fixture(scope="module", name="connection_kwargs")
def connection_kwargs_fixture():
    kwargs = {"hosts": [{"host": "localhost", "port": 9200, "scheme": "http"}]}
    yield kwargs


@pytest.fixture(scope="module", name="create_controller")
def create_controller_fixture(connection_kwargs):

    def inner_create_controller(*, connection_kwargs=connection_kwargs, refresh_period_secs=10, **controller_kwargs):
        return ElasticsearchController(connection_kwargs=connection_kwargs,
                                       refresh_period_secs=refresh_period_secs,
                                       **controller_kwargs)

    yield inner_create_controller


@pytest.mark.use_python
def test_constructor(create_controller: typing.Callable[..., ElasticsearchController], connection_kwargs: dict):
    assert create_controller(raise_on_exception=True)._raise_on_exception is True
    assert create_controller(refresh_period_secs=1.5)._refresh_period_secs == 1.5
    assert create_controller()._connection_kwargs == connection_kwargs


@pytest.mark.use_python
def test_apply_derive_params_func(create_controller: typing.Callable[..., ElasticsearchController]):
    pickled_func_str = str(pickle.dumps(mock_derive_params), encoding="latin1")

    # Create pickled_func_config
    pickled_func_config = {
        "pickled_func_str": pickled_func_str,  # Pickled function string
        "encoding": "latin1"
    }

    controller = create_controller()
    # Apply the mock function and check if connection_kwargs is updated
    controller._apply_derive_params_func(pickled_func_config)

    assert controller._connection_kwargs == {
        "hosts": [{
            "host": "localhost", "port": 9200, "scheme": "http"
        }], "retry_on_status": 3, "retry_on_timeout": 30
    }


@pytest.mark.use_python
def test_refresh_client_force(create_controller: typing.Callable[..., ElasticsearchController]):
    controller = create_controller(refresh_period_secs=1)

    client = controller._client
    is_refreshed = controller.refresh_client(force=True)

    controller._client.close.assert_called_once()
    assert client.ping.call_count == 2
    assert is_refreshed is True
    assert controller._last_refresh_time > 0


@pytest.mark.use_python
def test_refresh_client_not_needed(create_controller: typing.Callable[..., ElasticsearchController]):
    # Set last refresh time to a recent time
    current_time = time.time()
    controller = create_controller()
    controller._last_refresh_time = current_time
    client = controller._client

    # Simulate a refresh not needed scenario
    is_refreshed = controller.refresh_client()

    client.close.assert_not_called()
    assert client.ping.call_count == 1
    assert is_refreshed is False
    assert controller._last_refresh_time == current_time


@pytest.mark.use_python
def test_refresh_client_needed(create_controller: typing.Callable[..., ElasticsearchController]):

    # Set a 1 second refresh period
    controller = create_controller(refresh_period_secs=1)
    client = controller._client

    is_refreshed = False
    # Now "sleep" for more than 1 second to trigger a new client
    with patch("time.time", return_value=time.time() + 1):
        is_refreshed = controller.refresh_client()

    client.close.assert_called_once()
    assert client.ping.call_count == 2
    assert is_refreshed is True


@pytest.mark.use_python
@patch("morpheus.controllers.elasticsearch_controller.parallel_bulk", return_value=[(True, None)])
def test_parallel_bulk_write(mock_parallel_bulk, create_controller: typing.Callable[..., ElasticsearchController]):
    # Define your mock actions
    mock_actions = [{"_index": "test_index", "_id": 1, "_source": {"field1": "value1"}}]

    create_controller().parallel_bulk_write(actions=mock_actions)
    mock_parallel_bulk.assert_called_once()


def test_search_documents_success(create_controller: typing.Callable[..., ElasticsearchController]):
    controller = create_controller()
    controller._client.search.return_value = {"hits": {"total": 1, "hits": [{"_source": {"field1": "value1"}}]}}
    query = {"match": {"field1": "value1"}}
    result = controller.search_documents(index="test_index", query=query)

    assert isinstance(result, dict)
    assert "hits" in result
    assert "total" in result["hits"]
    assert result["hits"]["total"] == 1


def test_search_documents_failure_supress_errors(create_controller: typing.Callable[..., ElasticsearchController]):
    controller = create_controller()
    controller._client.search.side_effect = ConnectionError("Connection error")
    query = {"match": {"field1": "value1"}}
    result = controller.search_documents(index="test_index", query=query)

    assert isinstance(result, dict)
    assert not result


def test_search_documents_failure_raise_error(create_controller: typing.Callable[..., ElasticsearchController]):
    controller = create_controller(raise_on_exception=True)
    controller._client.search.side_effect = Exception
    query = {"match": {"field1": "value1"}}

    with pytest.raises(RuntimeError):
        controller.search_documents(index="test_index", query=query)
