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

import time

import mrc
import cudf
import tempfile
import os

import morpheus.modules.file_batcher  # Used to load and register morpheus modules
import morpheus.messages as messages


def on_next(control_msg):
    pass


def on_error():
    pass


def on_complete():
    pass


def test_contains_namespace():
    registry = mrc.ModuleRegistry

    assert registry.contains_namespace("morpheus")


def test_is_version_compatible():
    registry = mrc.ModuleRegistry

    release_version = [int(x) for x in mrc.__version__.split(".")]
    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True


def test_get_module():
    registry = mrc.ModuleRegistry

    fn_constructor = registry.get_module_constructor("FileBatcher", "morpheus")
    assert fn_constructor is not None

    config = {}
    module_instance = fn_constructor("ModuleFileBatcherTest", config)


packet_count = 1
packets_received = 0


def test_file_batcher_module(tmp_path):
    registry = mrc.ModuleRegistry

    fn_constructor = registry.get_module_constructor("FileBatcher", "morpheus")
    assert fn_constructor is not None

    input_filepaths = []
    filenames = [
        "TEST_2022-08-22T21_06_16.397Z.json",
        "TEST_2022-08-22T00_01_32.097Z.json",
        "TEST_2022-08-22T03_13_34.617Z.json",
        "TEST_2022-08-23T06_12_04.524Z.json",
        "TEST_2022-08-23T09_06_36.465Z.json",
        "TEST_2022-08-23T12_23_47.260Z.json",
        "TEST_2022-08-24T15_07_25.933Z.json",
        "TEST_2022-08-24T18_06_17.979Z.json",
        "TEST_2022-08-24T21_10_23.207Z.json"
    ]

    for filename in filenames:
        input_filepaths.append(os.path.join(tmp_path, filename))

    def init_wrapper(builder: mrc.Builder):

        df = cudf.DataFrame({
            'files': input_filepaths,
        }, columns=['files'])

        def gen_data():
            global packet_count
            config = {
                "tasks": [],
                "metadata": {
                    "data_type": "payload",
                    "batching_options": {
                        "start_time": "2022-08-20", "end_time": "2022-08-24", "period": "D", "sampling_rate_s": 0
                    }
                }
            }

            payload = messages.MessageMeta(df)
            msg = messages.MessageControl(config)
            msg.payload(payload)

            yield msg

        def _on_next(control_msg):
            global packets_received
            packets_received += 1
            assert (control_msg.payload().df == df)

        source = builder.make_source("source", gen_data)

        config = {
    "module_id": "FileBatcher", "module_name": "test_file_batcher", "namespace": "morpheus"
}
        # This will unpack the config and forward its payload (MessageMeta) to the sink
        file_batcher_module = builder.load_module("FileBatcher", "morpheus", "ModuleFileBatcherTest", config)

        sink = builder.make_sink("sink", _on_next, on_error, on_complete)

        builder.make_edge(source, file_batcher_module.input_port("input"))
        builder.make_edge(file_batcher_module.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert (packets_received == 3)

    for f in files:
        os.remove(f[0])


def test_file_loader_module():
    global packets_received
    packets_received = 0

    df = cudf.DataFrame(
        {
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': ['a', 'b', 'c', 'd', 'e'],
            'col4': [True, False, True, False, True]
        },
        columns=['col1', 'col2', 'col3', 'col4'])

    files = []
    file_types = ["csv", "parquet", "orc"]
    for ftype in file_types:
        _tempfile = tempfile.NamedTemporaryFile(suffix=f".{ftype}", delete=False)
        filename = _tempfile.name

        if ftype == "csv":
            df.to_csv(filename, index=False)
        elif ftype == "parquet":
            df.to_parquet(filename)
        elif ftype == "orc":
            df.to_orc(filename)

        files.append((filename, ftype))



if (__name__ == "__main__"):
    test_contains_namespace()
    test_is_version_compatible()
    test_get_module()
    test_file_batcher_module()
