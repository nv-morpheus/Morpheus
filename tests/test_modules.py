#!/usr/bin/env python
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

import pytest
import srf

from morpheus.modules.file_batcher_module import make_file_batcher_module
from morpheus.modules.file_to_df_module import make_file_to_df_module
from morpheus.modules.mlflow_model_writer_module import make_mlflow_model_writer_module

registry = srf.ModuleRegistry


@pytest.mark.use_python
def test_file_batcher_module(config):

    module_id = "TestFileBatcher"
    namespace = "test_morpheus_modules"

    make_file_batcher_module(module_id=module_id, namespace=namespace)

    assert registry.contains_namespace(namespace)
    assert registry.contains(module_id, namespace)

    registered_mod_dict = registry.registered_modules()

    assert "default" in registered_mod_dict
    assert module_id in registered_mod_dict[namespace]
    assert registry.contains(module_id, "default") is not True


@pytest.mark.use_python
def test_file_to_df_module():

    module_id = "TestFileToDataFrame"
    namespace = "test_morpheus_modules"

    make_file_to_df_module(module_id=module_id, namespace=namespace)

    assert registry.contains_namespace(namespace)
    assert registry.contains(module_id, namespace)

    registered_mod_dict = registry.registered_modules()

    assert "default" in registered_mod_dict
    assert module_id in registered_mod_dict[namespace]
    assert registry.contains(module_id, "default") is not True


@pytest.mark.use_python
def test_mlflow_model_writer_module():

    module_id = "TestMLFlowModelWriter"
    namespace = "test_morpheus_modules"

    make_mlflow_model_writer_module(module_id=module_id, namespace=namespace)

    assert registry.contains_namespace(namespace)
    assert registry.contains(module_id, namespace)

    registered_mod_dict = registry.registered_modules()

    assert "default" in registered_mod_dict
    assert module_id in registered_mod_dict[namespace]
    assert registry.contains(module_id, "default") is not True
