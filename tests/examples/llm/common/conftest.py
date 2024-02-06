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

from _utils import import_or_skip


@pytest.fixture(name="nemollm", autouse=True, scope='session')
def nemollm_fixture(fail_missing: bool):
    """
    All the tests in this subdir require nemollm
    """
    skip_reason = (
        "Tests for the WebScraperStage require the langchain package to be installed, to install this run:\n"
        "`conda env update --solver=libmamba -n morpheus "
        "--file morpheus/conda/environments/dev_cuda-121_arch-x86_64.yaml --prune`"
    )
    yield import_or_skip("langchain", reason=skip_reason, fail_missing=fail_missing)
