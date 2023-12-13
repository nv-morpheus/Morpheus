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

import pytest
import requests


@pytest.mark.slow
@pytest.mark.usefixtures("launch_mock_triton")
@pytest.mark.parametrize("iter_num", range(10))
def test_launch_camouflage(mock_rest_server: str, iter_num: int):
    rest_url = f"{mock_rest_server}/api/v1/data"
    triton_url = "http://localhost:8000/v2/health/live"

    resp = requests.get(rest_url)
    resp.raise_for_status()

    resp = requests.get(triton_url)
    resp.raise_for_status()
