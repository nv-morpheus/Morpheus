# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from _utils import require_env_variable


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
