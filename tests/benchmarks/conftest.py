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
import glob
import os
import random
import typing
from unittest import mock

import GPUtil
import pytest
from test_bench_e2e_pipelines import E2E_TEST_CONFIGS


# pylint: disable=unused-argument
def pytest_benchmark_update_json(config, benchmarks, output_json):
    gpus = GPUtil.getGPUs()

    for i, gpu in enumerate(gpus):
        # output_json["machine_info"]["gpu_" + str(i)] = gpu.name
        output_json["machine_info"]["gpu_" + str(i)] = {}
        output_json["machine_info"]["gpu_" + str(i)]["id"] = gpu.id
        output_json["machine_info"]["gpu_" + str(i)]["name"] = gpu.name
        output_json["machine_info"]["gpu_" + str(i)]["load"] = f"{gpu.load*100}%"
        output_json["machine_info"]["gpu_" + str(i)]["free_memory"] = f"{gpu.memoryFree}MB"
        output_json["machine_info"]["gpu_" + str(i)]["used_memory"] = f"{gpu.memoryUsed}MB"
        output_json["machine_info"]["gpu_" + str(i)]["temperature"] = f"{gpu.temperature} C"
        output_json["machine_info"]["gpu_" + str(i)]["uuid"] = gpu.uuid

    for bench in output_json['benchmarks']:
        if bench["name"] not in E2E_TEST_CONFIGS:
            continue

        line_count = 0
        byte_count = 0

        if "file_path" in E2E_TEST_CONFIGS[bench["name"]]:
            source_file = E2E_TEST_CONFIGS[bench["name"]]["file_path"]
            line_count = len(open(source_file, encoding='UTF-8').readlines())  # pylint: disable=consider-using-with
            byte_count = os.path.getsize(source_file)

        elif "input_glob_path" in E2E_TEST_CONFIGS[bench["name"]]:
            for file_name in glob.glob(E2E_TEST_CONFIGS[bench["name"]]["input_glob_path"]):
                line_count += len(open(file_name, encoding='UTF-8').readlines())  # pylint: disable=consider-using-with
                byte_count += os.path.getsize(file_name)

        repeat = E2E_TEST_CONFIGS[bench["name"]]["repeat"]

        bench["morpheus_config"] = {}
        bench["morpheus_config"]["num_threads"] = E2E_TEST_CONFIGS[bench["name"]]["num_threads"]
        bench["morpheus_config"]["pipeline_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["pipeline_batch_size"]
        bench["morpheus_config"]["model_max_batch_size"] = E2E_TEST_CONFIGS[bench["name"]]["model_max_batch_size"]
        bench["morpheus_config"]["feature_length"] = E2E_TEST_CONFIGS[bench["name"]]["feature_length"]
        bench["morpheus_config"]["edge_buffer_size"] = E2E_TEST_CONFIGS[bench["name"]]["edge_buffer_size"]

        bench['stats']["input_lines"] = line_count * repeat
        bench['stats']['min_throughput_lines'] = (line_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_lines'] = (line_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_lines'] = (line_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_lines'] = (line_count * repeat) / bench['stats']['median']
        bench['stats']["input_bytes"] = byte_count * repeat
        bench['stats']['min_throughput_bytes'] = (byte_count * repeat) / bench['stats']['max']
        bench['stats']['max_throughput_bytes'] = (byte_count * repeat) / bench['stats']['min']
        bench['stats']['mean_throughput_bytes'] = (byte_count * repeat) / bench['stats']['mean']
        bench['stats']['median_throughput_bytes'] = (byte_count * repeat) / bench['stats']['median']


@pytest.mark.usefixtures("openai")
@pytest.fixture(name="mock_chat_completion")
@pytest.mark.usefixtures()
def mock_chat_completion_fixture(mock_chat_completion: mock.MagicMock):

    async def sleep_first(*args, **kwargs):
        # Sleep time is based on average request time
        await asyncio.sleep(1.265)
        return mock.DEFAULT

    mock_chat_completion.acreate.side_effect = sleep_first

    yield mock_chat_completion


@pytest.mark.usefixtures("nemollm")
@pytest.fixture(name="mock_nemollm")
def mock_nemollm_fixture(mock_nemollm: mock.MagicMock):
    # The generate function is a blocking call that returns a future when return_type="async"

    async def sleep_first(fut: asyncio.Future, value: typing.Any = mock.DEFAULT):
        # Sleep time is based on average request time
        await asyncio.sleep(0.412)
        fut.set_result(value)

    def create_future(*args, **kwargs) -> asyncio.Future:
        event_loop = asyncio.get_event_loop()
        fut = event_loop.create_future()
        event_loop.create_task(sleep_first(fut, mock.DEFAULT))
        return fut

    mock_nemollm.generate.side_effect = create_future

    yield mock_nemollm
