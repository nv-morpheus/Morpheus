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

import os
import types
from unittest import mock

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.stages.input.arxiv_source import ArxivSource


@pytest.fixture(name="pdf_file")
def pdf_file_fixture():
    yield os.path.join(TEST_DIRS.tests_data_dir, "test.pdf")


def test_constructor(config: Config):
    cache_dir = "/does/not/exist"
    stage = ArxivSource(config, query="unittest", cache_dir=cache_dir)
    assert stage._query == "unittest"
    assert stage._cache_dir == cache_dir

    assert not os.path.exists(cache_dir)


def _make_mock_result(file_name: str):
    result = mock.MagicMock()
    result._get_default_filename.return_value = file_name
    return result


@pytest.mark.parametrize("use_subdir", [False, True])
def test_generate_frames_cache_miss(mock_arxiv_search: mock.MagicMock, config: Config, tmp_path: str, use_subdir: bool):
    if use_subdir:
        # Tests that the cache directory is created if it doesn't exist
        cache_dir = os.path.join(tmp_path, "cache")
        assert not os.path.exists(cache_dir)
    else:
        cache_dir = tmp_path

    stage = ArxivSource(config, query="unittest", cache_dir=cache_dir)

    expected_file_paths = [os.path.join(cache_dir, "apples.pdf"), os.path.join(cache_dir, "plums.pdf")]
    assert list(stage._generate_frames()) == expected_file_paths

    assert os.path.exists(cache_dir)
    assert stage._total_pdfs == 2

    mock_arxiv_search.assert_called_once()
    mock_arxiv_search.results.assert_called_once()
    mock_results: list[mock.MagicMock] = mock_arxiv_search.results.return_value
    for mock_result in mock_results:
        mock_result.download_pdf.assert_called_once()


def test_generate_frames_cache_hit(mock_arxiv_search: mock.MagicMock, config: Config, tmp_path: str):
    with open(os.path.join(tmp_path, "apples.pdf"), "w", encoding="utf-8") as f:
        f.write("apples")

    stage = ArxivSource(config, query="unittest", cache_dir=tmp_path)

    expected_file_paths = [os.path.join(tmp_path, "apples.pdf"), os.path.join(tmp_path, "plums.pdf")]
    assert list(stage._generate_frames()) == expected_file_paths

    assert stage._total_pdfs == 2

    mock_arxiv_search.assert_called_once()
    mock_arxiv_search.results.assert_called_once()
    mock_results: list[mock.MagicMock] = mock_arxiv_search.results.return_value
    for mock_result in mock_results:
        if mock_result._get_default_filename.return_value == "apples.pdf":
            mock_result.download_pdf.assert_not_called()
        else:
            mock_result.download_pdf.assert_called_once()


def test_process_pages(mock_arxiv_search: mock.MagicMock, config: Config, pdf_file: str):
    stage = ArxivSource(config, query="unittest", cache_dir="/does/not/exist")
    documents = stage._process_pages(pdf_file)

    mock_arxiv_search.assert_not_called()
    assert stage._total_pages == 1

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == 'Morpheus\nunittest'
    assert document.metadata == {'source': pdf_file, 'page': 0}


def test_process_pages_error(mock_arxiv_search: mock.MagicMock, config: Config, tmp_path: str):
    bad_pdf_filename = os.path.join(tmp_path, "bad.pdf")
    with open(bad_pdf_filename, 'w', encoding='utf-8') as fh:
        fh.write("Not a PDF")

    stage = ArxivSource(config, query="unittest", cache_dir=tmp_path)

    with pytest.raises(RuntimeError):
        stage._process_pages(bad_pdf_filename)

    mock_arxiv_search.assert_not_called()


@mock.patch("langchain.document_loaders.PyPDFLoader")
def test_process_pages_retry(mock_pdf_loader: mock.MagicMock,
                             mock_arxiv_search: mock.MagicMock,
                             config: Config,
                             pypdf: types.ModuleType):
    call_count = 0

    def mock_load():
        nonlocal call_count
        call_count += 1
        if call_count < 5:
            raise pypdf.errors.PdfStreamError()
        elif call_count == 5:
            return ["unittest"]

        assert False, "Should not be called more than 5 times"

    mock_pdf_loader.return_value = mock_pdf_loader
    mock_pdf_loader.load.side_effect = mock_load
    stage = ArxivSource(config, query="unittest", cache_dir="/does/not/exist")
    documents = stage._process_pages('/does/not/exist/fake.pdf')

    mock_arxiv_search.assert_not_called()
    assert stage._total_pages == 1
    assert call_count == 5

    assert documents == ["unittest"]
