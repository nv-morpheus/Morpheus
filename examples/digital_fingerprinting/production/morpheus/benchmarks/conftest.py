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

import glob
import json
from os import path

import GPUtil

from benchmarks.test_bench_e2e_dfp_pipeline import PIPELINES_CONF


def get_json_lines_count(filename):
    with open(filename, 'r') as f:
        lines = json.loads(f.read())
    return len(lines)


def pytest_benchmark_update_json(config, benchmarks, output_json):

    curr_dir = path.dirname(path.abspath(__file__))

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

    line_count = 0
    byte_count = 0

    for bench in output_json['benchmarks']:
        if "file_path" in PIPELINES_CONF[bench["name"]]:
            source_file = PIPELINES_CONF[bench["name"]]["file_path"]
            source_file = path.join(curr_dir, source_file)
            line_count = get_json_lines_count(source_file)
            byte_count = path.getsize(source_file)

        elif "glob_path" in PIPELINES_CONF[bench["name"]]:
            source_files_glob = path.join(curr_dir, PIPELINES_CONF[bench["name"]]["glob_path"])
            for fn in glob.glob(source_files_glob):
                line_count += get_json_lines_count(fn)
                byte_count += path.getsize(fn)
        else:
            raise KeyError("Configuration requires either 'glob_path' or 'file_path' attribute.")

        bench["morpheus_config"] = {}
        bench["morpheus_config"]["num_threads"] = PIPELINES_CONF[bench["name"]]["num_threads"]
        bench["morpheus_config"]["pipeline_batch_size"] = PIPELINES_CONF[bench["name"]]["pipeline_batch_size"]
        bench["morpheus_config"]["edge_buffer_size"] = PIPELINES_CONF[bench["name"]]["edge_buffer_size"]
        bench["morpheus_config"]["start_time"] = PIPELINES_CONF[bench["name"]]["start_time"]
        bench["morpheus_config"]["duration"] = PIPELINES_CONF[bench["name"]]["duration"]

        bench['stats']["input_lines"] = line_count
        bench['stats']['min_throughput_lines'] = line_count / bench['stats']['max']
        bench['stats']['max_throughput_lines'] = line_count / bench['stats']['min']
        bench['stats']['mean_throughput_lines'] = line_count / bench['stats']['mean']
        bench['stats']['median_throughput_lines'] = line_count / bench['stats']['median']
        bench['stats']["input_bytes"] = byte_count
        bench['stats']['min_throughput_bytes'] = byte_count / bench['stats']['max']
        bench['stats']['max_throughput_bytes'] = byte_count / bench['stats']['min']
        bench['stats']['mean_throughput_bytes'] = byte_count / bench['stats']['mean']
        bench['stats']['median_throughput_bytes'] = byte_count / bench['stats']['median']
