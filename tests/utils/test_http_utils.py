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

from unittest import mock

import pytest

from morpheus.utils import http_utils


@pytest.mark.parametrize("url,expected",
                         [("nvidia.com/", "http://nvidia.com/"), ("https://nvidia.com", "https://nvidia.com"),
                          ("http://nvidia.com", "http://nvidia.com"),
                          ("localhost:8080/a/b/c?test=this", "http://localhost:8080/a/b/c?test=this"),
                          ("ftp://tester@nvidia.com/robots.txt", "ftp://tester@nvidia.com/robots.txt")])
def test_prepare_url(url: str, expected: str):
    assert http_utils.prepare_url(url) == expected


def test_prepare_url_error():
    with pytest.raises(ValueError):
        http_utils.prepare_url("")


def make_mock_response(mock_request_session: mock.MagicMock,
                       status_code: int = 200,
                       content_type: str = http_utils.MimeTypes.TEXT.value,
                       text: str = "test"):
    mock_response = mock.MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {"Content-Type": content_type}
    mock_response.text = text

    mock_request_session.return_value = mock_request_session
    mock_request_session.request.return_value = mock_response
    return mock_response


@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_request_with_retry(mock_sleep: mock.MagicMock, mock_request_session: mock.MagicMock):
    mock_response = make_mock_response(mock_request_session)

    response = http_utils.request_with_retry({'method': 'GET', 'url': 'http://test.nvidia.com'})

    assert response == (mock_request_session, mock_response)
    mock_request_session.request.assert_called_once_with(method='GET', url='http://test.nvidia.com')
    mock_sleep.assert_not_called()


@pytest.mark.parametrize("respect_retry_after_header", [True, False])
@mock.patch("requests.Session")
@mock.patch("time.sleep")
def test_request_with_retry_max_retries(mock_sleep: mock.MagicMock,
                                        mock_request_session: mock.MagicMock,
                                        respect_retry_after_header: bool):
    mock_response = make_mock_response(mock_request_session, status_code=500)
    mock_response.headers = {"Retry-After": "42"} if respect_retry_after_header else {}

    with pytest.raises(RuntimeError):
        http_utils.request_with_retry({'method': 'GET', 'url': 'http://test.nvidia.com'}, max_retries=5, sleep_time=1)

    assert mock_request_session.request.call_count == 5
    assert mock_sleep.call_count == 4
    if respect_retry_after_header:
        mock_sleep.assert_has_calls([mock.call(42)] * 4)
    else:
        mock_sleep.assert_has_calls([
            mock.call(1),
            mock.call(2),
            mock.call(4),
            mock.call(8),
        ])
