# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import importlib
import logging
import os
import subprocess
import time

import pytest
import requests
from test_bench_e2e_pipelines import TEST_SOURCES


def pytest_benchmark_update_json(config, benchmarks, output_json):
    line_count = 0
    byte_count = 0
    for bench in output_json['benchmarks']:
        if "file_path" in TEST_SOURCES[bench["name"]]:
            source_file = TEST_SOURCES[bench["name"]]["file_path"]
            line_count = len(open(source_file).readlines())
            byte_count = os.path.getsize(source_file)

        elif "glob_path" in TEST_SOURCES[bench["name"]]:
            for fn in glob.glob(TEST_SOURCES[bench["name"]]["glob_path"]):
                line_count += len(open(fn).readlines())
                byte_count += os.path.getsize(fn)

        repeat = TEST_SOURCES[bench["name"]]["repeat"]

        bench['stats']['min-throughput-lines'] =  (line_count*repeat) / bench['stats']['max']
        bench['stats']['max-throughput-lines'] =  (line_count*repeat) / bench['stats']['min']
        bench['stats']['mean-throughput-lines'] =  (line_count*repeat) / bench['stats']['mean']
        bench['stats']['median-throughput-lines'] =  (line_count*repeat) / bench['stats']['median']
        bench['stats']['min-throughput-bytes'] =  (byte_count*repeat) / bench['stats']['max']
        bench['stats']['max-throughput-bytes'] =  (byte_count*repeat) / bench['stats']['min']
        bench['stats']['mean-throughput-bytes'] =  (byte_count*repeat) / bench['stats']['mean']
        bench['stats']['median-throughput-bytes'] =  (byte_count*repeat) / bench['stats']['median']
