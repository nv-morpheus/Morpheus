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
from _utils import require_env_variable


@pytest.fixture(name="nemollm", scope='session')
def nemollm_fixture(fail_missing: bool):
    """
    Fixture to ensure nemollm is installed
    """
    skip_reason = ("Tests for the NeMoLLMService require the nemollm package to be installed, to install this run:\n"
                   "`mamba install -n base -c conda-forge conda-merge`\n"
                   "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
                   "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
                   "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")
    yield import_or_skip("nemollm", reason=skip_reason, fail_missing=fail_missing)


@pytest.fixture(name="openai", scope='session')
def openai_fixture(fail_missing: bool):
    """
    Fixture to ensure openai is installed
    """
    skip_reason = ("Tests for the OpenAIChatService require the openai package to be installed, to install this run:\n"
                   "`mamba install -n base -c conda-forge conda-merge`\n"
                   "`conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml "
                   "  docker/conda/environments/cuda${CUDA_VER}_examples.yml"
                   "  > .tmp/merged.yml && mamba env update -n morpheus --file .tmp/merged.yml`")
    yield import_or_skip("openai", reason=skip_reason, fail_missing=fail_missing)


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_chat_completion")
def mock_chat_completion_fixture():
    with mock.patch("openai.ChatCompletion") as mock_chat_completion:
        mock_chat_completion.return_value = mock_chat_completion

        response = {'choices': [{'message': {'content': 'test_output'}}]}
        mock_chat_completion.create.return_value = response.copy()
        mock_chat_completion.acreate = mock.AsyncMock(return_value=response.copy())
        yield mock_chat_completion


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm")
def mock_nemollm_fixture():
    with mock.patch("nemollm.NemoLLM") as mock_nemollm:
        mock_nemollm.return_value = mock_nemollm
        mock_nemollm.generate_multiple.return_value = ["test_output"]
        mock_nemollm.post_process_generate_response.return_value = {"text": "test_output"}

        yield mock_nemollm


@pytest.fixture(name="countries")
def countries_fixture():
    yield [
        "Eldoria",
        "Drakoria",
        "Avaloria",
        "Mystralia",
        "Faerundor",
        "Glimmerholme",
        "Emberfell",
        "Stormhaven",
        "Frosthold",
        "Celestria"
    ]


@pytest.fixture(name="capitals")
def capitals_fixture():
    yield [
        "Thundertop",
        "Dragonspire",
        "Starhaven",
        "Enigma Citadel",
        "Moonshroud",
        "Silverglade",
        "Infernix",
        "Skyreach",
        "Icicle Keep",
        "Seraphia"
    ]


@pytest.fixture(name="country_prompts")
def country_prompts_fixture(countries: list[str]):
    yield [f"What is the capital of {country}?" for country in countries]


@pytest.fixture(name="capital_responses")
def capital_responses_fixture(countries: list[str], capitals: list[str]):
    assert len(countries) == len(capitals)

    responses = []
    for (i, country) in enumerate(countries):
        responses.append(f"The capital of {country} is {capitals[i]}.")

    yield responses


@pytest.fixture(name="ngc_api_key", scope='session')
def ngc_api_key_fixture():
    """
    Integration tests require an NGC API key.
    """
    yield require_env_variable(
        varname="NGC_API_KEY",
        reason="nemo integration tests require the `NGC_API_KEY` environment variable to be defined.")


@pytest.fixture(name="openai_api_key", scope='session')
def openai_api_key_fixture():
    """
    Integration tests require an Openai API key.
    """
    yield require_env_variable(
        varname="OPENAI_API_KEY",
        reason="openai integration tests require the `OPENAI_API_KEY` environment variable to be defined.")


@pytest.fixture(name="serpapi_api_key", scope='session')
def serpapi_api_key_fixture():
    """
    Integration tests require a Serpapi API key.
    """
    yield require_env_variable(
        varname="SERPAPI_API_KEY",
        reason="serpapi integration tests require the `SERPAPI_API_KEY` environment variable to be defined.")
