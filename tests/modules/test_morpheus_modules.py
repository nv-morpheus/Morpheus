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

import time

import mrc

import morpheus._lib.messages as _messages
import morpheus.modules  # Used to load and register morpheus modules
import morpheus.messages as messages


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

    fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
    assert fn_constructor is not None

    config = {}
    module_instance = fn_constructor("ModuleDataLoaderTest", config)


def test_init_module():
    def init_wrapper(builder: mrc.Builder):
        def gen_data():
            config = {"loader_id": "payload"}
            for i in range(10):
                yield messages.MessageControl(config)

        def on_next(data):
            pass

        def on_error():
            pass

        def on_complete():
            pass

        registry = mrc.ModuleRegistry

        fn_constructor = registry.get_module_constructor("DataLoader", "morpheus")
        assert fn_constructor is not None

        source = builder.make_source("source", gen_data)

        config = {}
        data_loader = builder.load_module("DataLoader", "morpheus", "ModuleDataLoaderTest", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source, data_loader.input_port("input"))
        builder.make_edge(data_loader.output_port("output"), sink)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("main", init_wrapper)

    options = mrc.Options()
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()


if (__name__ == "__main__"):
    test_contains_namespace()
    test_is_version_compatible()
    test_get_module()
    test_init_module()
