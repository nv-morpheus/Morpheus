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

from unittest import mock

import pytest

from _utils import import_or_skip


@pytest.fixture(name="nemollm", autouse=True, scope='session')
def nemollm_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require nemollm
    """
    skip_reason = ("Tests for the NeMoLLMService require the nemollm package to be installed, to install this run:\n"
                   "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")
    yield import_or_skip("nemollm", reason=skip_reason, fail_missing=fail_missing)


@pytest.fixture(name="openai", autouse=True, scope='session')
def openai_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require openai
    """
    skip_reason = ("Tests for the OpenAIChatService require the openai package to be installed, to install this run:\n"
                   "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")
    yield import_or_skip("openai", reason=skip_reason, fail_missing=fail_missing)


# Using autouse to ensure we never attempt to actually call either of these services
@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_chat_completion", autouse=True)
def mock_chat_completion_fixture():
    with mock.patch("openai.ChatCompletion") as mock_chat_completion:
        mock_chat_completion.return_value = mock_chat_completion

        response = {'choices': [{'message': {'content': 'test_output'}}]}
        mock_chat_completion.create.return_value = response.copy()
        mock_chat_completion.acreate = mock.AsyncMock(return_value=response.copy())
        yield mock_chat_completion


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm", autouse=True)
def mock_nemollm_fixture():
    with mock.patch("nemollm.NemoLLM") as mock_nemollm:
        mock_nemollm.return_value = mock_nemollm
        mock_nemollm.generate_multiple.return_value = ["test_output"]
        mock_nemollm.post_process_generate_response.return_value = {"text": "test_output"}

        yield mock_nemollm


@pytest.fixture(name="mock_nemo_service")
def mock_nemo_service_fixture(mock_nemollm: mock.MagicMock):
    mock_nemo_service = mock.MagicMock()
    mock_nemo_service.return_value = mock_nemo_service
    mock_nemo_service._conn = mock_nemollm
    return mock_nemo_service
