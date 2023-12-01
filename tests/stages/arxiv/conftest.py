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

import os
from unittest import mock

import pytest

from _utils import import_or_skip

SKIP_REASON = ("Tests for the arxiv_source require a number of packages not installed in the Morpheus development "
               "environment. To install these run:\n"
               "`mamba install -n base -c conda-forge conda-merge`\n"
               "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
               "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
               "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")


@pytest.fixture(name="arxiv", autouse=True, scope='session')
def arxiv_fixture(fail_missing: bool):
    """
    All the tests in this subdir require arxiv
    """
    yield import_or_skip("arxiv", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(name="langchain", autouse=True, scope='session')
def langchain_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require langchain
    """
    yield import_or_skip("langchain", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(name="pypdf", autouse=True, scope='session')
def pypdf_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require pypdf
    """
    yield import_or_skip("pypdf", reason=SKIP_REASON, fail_missing=fail_missing)


def _make_mock_result(file_name: str):
    result = mock.MagicMock()
    result._get_default_filename.return_value = file_name
    return result


@pytest.fixture(name="mock_arxiv_search")
def mock_arxiv_search_fixture():
    """
    Mocks the arxiv search function to prevent tests from performing actual searches.
    """
    with mock.patch("arxiv.Search") as mock_search:
        mock_search.return_value = mock_search
        mock_search.results.return_value = [_make_mock_result("apples.pdf"), _make_mock_result("plums.pdf")]
        yield mock_search


@pytest.fixture(name="pdf_file")
def pdf_file_fixture():
    from _utils import TEST_DIRS
    yield os.path.join(TEST_DIRS.tests_data_dir, "test.pdf")
