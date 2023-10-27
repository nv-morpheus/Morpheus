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

import pytest

from _utils import import_or_skip
from _utils import require_env_variable


@pytest.fixture(name="nemollm", autouse=True, scope='session')
def nemollm_fixture(fail_missing: bool):
    """
    All of the tests in this subdir require nemollm
    """
    skip_reason = ("Tests for the NeMoLLMService require the nemollm package to be installed, to install this run:\n"
                   "`mamba env update -n ${CONDA_DEFAULT_ENV} --file docker/conda/environments/cuda11.8_examples.yml`")
    yield import_or_skip("nemollm", reason=skip_reason, fail_missing=fail_missing)


@pytest.fixture(name="ngc_api_key", scope='session')
def ngc_api_key_fixture(fail_missing: bool):
    """
    Integration tests require an NGC API key.
    """
    yield require_env_variable(
        varname="NGC_API_KEY",
        reason="nemo integration tests require the `NGC_API_KEY` environment variavble to be defined.",
        fail_missing=fail_missing)
