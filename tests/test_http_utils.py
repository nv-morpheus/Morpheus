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
