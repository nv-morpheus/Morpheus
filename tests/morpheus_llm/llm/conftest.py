# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types

import pytest

from _utils import require_env_variable


@pytest.fixture(name="openai", scope='session', autouse=True)
def openai_fixture(openai: types.ModuleType):
    """
    Fixture to ensure openai is installed
    """
    yield openai


@pytest.fixture(name="langchain", scope='session', autouse=True)
def langchain_fixture(langchain: types.ModuleType):
    """
    Fixture to ensure langchain is installed
    """
    yield langchain


@pytest.fixture(name="langchain_core", scope='session', autouse=True)
def langchain_core_fixture(langchain_core: types.ModuleType):
    """
    Fixture to ensure langchain_core is installed
    """
    yield langchain_core


@pytest.fixture(name="langchain_community", scope='session', autouse=True)
def langchain_community_fixture(langchain_community: types.ModuleType):
    """
    Fixture to ensure langchain_community is installed
    """
    yield langchain_community


@pytest.fixture(name="langchain_openai", scope='session', autouse=True)
def langchain_openai_fixture(langchain_openai: types.ModuleType):
    """
    Fixture to ensure langchain_openai is installed
    """
    yield langchain_openai


@pytest.fixture(name="langchain_nvidia_ai_endpoints", scope='session', autouse=True)
def langchain_nvidia_ai_endpoints_fixture(langchain_nvidia_ai_endpoints: types.ModuleType):
    """
    Fixture to ensure langchain_nvidia_ai_endpoints is installed
    """
    yield langchain_nvidia_ai_endpoints


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
