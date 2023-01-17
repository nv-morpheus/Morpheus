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

from unittest.mock import MagicMock

import pytest
from requests.models import Response

from morpheus.pipeline.training.tao_client import TaoApiClient
from morpheus.pipeline.training.tao_client import generate_schema_url
from morpheus.pipeline.training.tao_client import vaildate_apikey


def test_generate_schema_url():

    actual = generate_schema_url(url="localhost:32080", ssl=False)
    expected = "http://localhost:32080"

    assert actual == expected

    with pytest.raises(ValueError):
        generate_schema_url(url="http://localhost:32080", ssl=False)

    with pytest.raises(ValueError):
        generate_schema_url(url="https://localhost:32080", ssl=True)

    actual = generate_schema_url(url="localhost:32080", ssl=False)
    expected = "http://localhost:32080"
    assert actual == expected

    actual = generate_schema_url(url="localhost:32080", ssl=True)
    expected = "https://localhost:32080"
    assert actual == expected


def test_vaildate_apikey():

    vaildate_apikey("test_api_key")

    with pytest.raises(ValueError):
        vaildate_apikey("")

    with pytest.raises(ValueError):
        vaildate_apikey(123459)


def test_create_resource():
    tao_client = TaoApiClient("test_api_key", "localhost:32080")

    mock_creds = {"user_id": "X20109876", "token": "TOkJJTkw6WkxKRDpNWk9ZOkRVN0o6"}

    mock_response = Response()
    mock_response.status_code = 201
    mock_response._content = b'{ "id" : "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx", "created_on": "2023-01-17T15:35:08.014463"}'

    tao_client.get_login_creds = MagicMock(return_value=mock_creds)
    tao_client.session.post = MagicMock(return_value=Response())
    ds_type = "object_detection"
    ds_format = "kitti"

    data = {"type": ds_type, "format": ds_format}

    actual_resource_id = tao_client.create_resource(kind="dataset", data=data)

    assert actual_resource_id == "eyJzdWIiOiJwOTltOTh0NzBzdDFsa3Zx"
