# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from http import HTTPStatus
from unittest import mock

import pytest
import requests

from morpheus.common import RestServer
from morpheus.utils.http_utils import MimeTypes


def make_parse_fn(status: HTTPStatus = HTTPStatus.OK,
                  content_type: str = "unit/test",
                  content: str = 'TEST OK',
                  on_complete_cb: typing.Callable = None) -> mock.MagicMock:
    mock_parse_fn = mock.MagicMock()
    mock_parse_fn.return_value = (status.value, content_type, content, on_complete_cb)
    return mock_parse_fn


def make_url(port: int, endpoint: str):
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return f"http://127.0.0.1:{port}{endpoint}"


@pytest.mark.parametrize("endpoint", ["/test", "test/", "/test/", "/a/b/c/d"])
@pytest.mark.parametrize("port", [8080, 8090, 9090])
@pytest.mark.parametrize("method", ["GET", "POST", "PUT"])
@pytest.mark.parametrize("use_callback", [True, False])
@pytest.mark.parametrize("use_context_mgr", [True, False])
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("status,content_type,content",
                         [(HTTPStatus.OK, MimeTypes.TEXT.value, "OK"),
                          (HTTPStatus.OK, MimeTypes.JSON.value, '{"test": "OK"}'),
                          (HTTPStatus.NOT_FOUND, MimeTypes.TEXT.value, "NOT FOUND"),
                          (HTTPStatus.INTERNAL_SERVER_ERROR, MimeTypes.TEXT.value, "Unexpected error")])
def test_simple_request(port: int,
                        endpoint: str,
                        method: str,
                        status: HTTPStatus,
                        content_type: str,
                        content: str,
                        use_callback: bool,
                        use_context_mgr: bool,
                        num_threads: int):
    if use_callback:
        mock_callback = mock.MagicMock()
    else:
        mock_callback = None

    parse_fn = make_parse_fn(status=status, content_type=content_type, content=content, on_complete_cb=mock_callback)

    url = make_url(port, endpoint)
    if method == "GET":
        payload = ''
    else:
        if content_type == MimeTypes.JSON.value:
            payload = '{"test": "this"}'
        else:
            payload = "test"

    server = None

    def check_server():
        print("Checking server", flush=True)
        assert server.is_running()

        print("CS Req", flush=True)
        response = requests.request(method=method, url=url, data=payload, timeout=1.0)

        print("CS Resp", flush=True)
        assert response.status_code == status.value
        assert response.headers["Content-Type"] == content_type
        assert response.text == content

        parse_fn.assert_called_once_with(payload)
        if use_callback:
            mock_callback.assert_called_once_with(False, "")

        print("CS Done", flush=True)

    if use_context_mgr:
        print("Using context manager", flush=True)
        with RestServer(parse_fn=parse_fn, port=port, endpoint=endpoint, method=method,
                        num_threads=num_threads) as server:
            print("CM", flush=True)
            assert server.is_running()
            print("CM Start", flush=True)
            check_server()

    else:
        server = RestServer(parse_fn=parse_fn, port=port, endpoint=endpoint, method=method, num_threads=num_threads)
        assert not server.is_running()
        server.start()

        check_server()

        server.stop()

    assert not server.is_running()
