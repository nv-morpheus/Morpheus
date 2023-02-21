#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from morpheus.stages.inference.triton_inference_stage import ResourcePool


@pytest.mark.use_python
def test_resource_pool():
    create_fn = mock.MagicMock()

    # If called a third time this will raise a StopIteration exception
    create_fn.side_effect = range(2)

    rp = ResourcePool(create_fn)
    assert rp._queue.qsize() == 0

    assert rp.borrow() == 0
    assert rp._queue.qsize() == 0
    assert rp._added_count == 1
    create_fn.assert_called_once()

    assert rp.borrow() == 1
    assert rp._queue.qsize() == 0
    assert rp._added_count == 2
    assert create_fn.call_count == 2

    rp.return_obj(0)
    assert rp._queue.qsize() == 1
    rp.return_obj(1)
    assert rp._queue.qsize() == 2

    assert rp.borrow() == 0
    assert rp._queue.qsize() == 1
    assert rp._added_count == 2
    assert create_fn.call_count == 2

    assert rp.borrow() == 1
    assert rp._queue.qsize() == 0
    assert rp._added_count == 2
    assert create_fn.call_count == 2
