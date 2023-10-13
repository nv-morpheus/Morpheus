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

import asyncio

from morpheus._lib import pycoro  # pylint: disable=morpheus-incorrect-lib-from-import


async def test_pycoro():

    hit_inside = False

    async def inner():

        nonlocal hit_inside

        result = await pycoro.wrap_coroutine(asyncio.sleep(1, result=['a', 'b', 'c']))

        hit_inside = True

        return [result]

    returned_val = await pycoro.wrap_coroutine(inner())

    assert returned_val == 'a'
    assert hit_inside


async def test_pycoro_many():

    expected_count = 1000
    hit_count = 0

    start_time = asyncio.get_running_loop().time()

    async def inner():

        nonlocal hit_count

        await asyncio.sleep(1)

        hit_count += 1

        return ['a', 'b', 'c']

    coros = [pycoro.wrap_coroutine(inner()) for _ in range(expected_count)]

    returned_vals = await asyncio.gather(*coros)

    end_time = asyncio.get_running_loop().time()

    assert returned_vals == ['a'] * expected_count
    assert hit_count == expected_count
    assert (end_time - start_time) < 1.5
