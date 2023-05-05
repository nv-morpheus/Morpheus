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

import mrc
import pytest

import cudf

import morpheus.messages as messages
# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import morpheus.modules  # noqa: F401
from morpheus.messages import ControlMessage

PACKET_COUNT = 5


def on_next(data):
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

    fn_constructor = registry.get_module_constructor("ToControlMessage", "morpheus")
    assert fn_constructor is not None

    config = {}
    module_instance = fn_constructor("ToControlMessageTest", config)  # noqa: F841 -- we don't need to use it


def test_get_module_with_bad_config():

    def init_wrapper(builder: mrc.Builder):

        def gen_data():
            for i in range(PACKET_COUNT):
                yield i

        source = builder.make_source("source", gen_data)

        config = {"meta_data": {"data_type": "streaming"}}
        to_control_message = builder.load_module("ToControlMessage", "morpheus", "ToControlMessageTest", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source, to_control_message.input_port("input"))
        builder.make_edge(to_control_message.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)

    # This should fail, because no valid module config is specified
    with pytest.raises(ValueError):
        executor.start()
        executor.join()


def test_to_control_message_module():
    packets_received = 0
    registry = mrc.ModuleRegistry
    fn_constructor = registry.get_module_constructor("Multiplexer", "morpheus")
    assert fn_constructor is not None

    def init_wrapper(builder: mrc.Builder):

        df = cudf.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': ['a', 'b', 'c', 'd', 'e'],
            'col4': [True, False, True, False, True]
        })

        def gen_data():
            global PACKET_COUNT

            meta = messages.MessageMeta(df)

            for i in range(PACKET_COUNT):
                yield meta

        def _on_next(control_msg: ControlMessage):
            nonlocal packets_received
            packets_received += 1
            assert (control_msg.payload().df == df)

        source = builder.make_source("source", gen_data)

        config = {"meta_data": {"data_type": "streaming"}, "tasks": [{"type": "inference", "properties": {}}]}
        to_control_message = builder.load_module("ToControlMessage", "morpheus", "ToControlMessageTest", config)

        sink = builder.make_sink("sink", _on_next, on_error, on_complete)

        builder.make_edge(source, to_control_message.input_port("input"))
        builder.make_edge(to_control_message.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()

    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()
    # We have 2 sources that are generating packets.
    assert packets_received == PACKET_COUNT


if (__name__ == "__main__"):
    test_contains_namespace()
    test_is_version_compatible()
    test_get_module()
    test_get_module_with_bad_config()
    test_to_control_message_module()
