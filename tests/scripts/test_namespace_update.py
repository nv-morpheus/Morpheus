#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import os

import pytest

from _utils import TEST_DIRS


@pytest.fixture(name="scripts_data_dir", scope="module")
def scripts_data_dir_fixture(tmp_path) -> str:
    '''
    Copy the data to a temporary directory as we will be modifying the files.
    '''
    data_dir = os.path.join(TEST_DIRS.tests_dir, "scripts/data")
    tmp_data_dir = tmp_path / "scripts"
    tmp_data_dir.mkdir()
    os.system(f"cp -r {data_dir} {tmp_data_dir}")
    return os.path.join(tmp_data_dir, "data")


@pytest.fixture(name="update_namespace_script", scope="module")
def update_namespace_script_fixture() -> str:
    return os.path.join(TEST_DIRS.morpheus_root, "scripts/morpheus_namespace_update.py")


def import_module_from_path(module_name, path) -> tuple:
    '''
    Import a module from the pytest tmp_path.
    '''
    # Create a module spec from the given path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec:
        return None, None

    # Load the module from the created spec
    module = importlib.util.module_from_spec(spec)
    if not module:
        return None, None

    return spec, module


@pytest.mark.usefixtures("morpheus_dfp")
def test_dfp_namespace_update(scripts_data_dir, update_namespace_script):
    '''
    Update the DFP namespace imports and verify the imports work.
    '''
    module_name = 'dfp_old_namespace_data'
    module_path = os.path.join(scripts_data_dir, f'{module_name}.py')

    # dfp imports expected to fail before namespace update
    spec, module = import_module_from_path(module_name, module_path)
    assert module is not None, f"Failed to import {module_name} from {module_path}"
    try:
        spec.loader.exec_module(module)
        assert False, "dfp_namespace_data input is not setup with the old imports"
    except ModuleNotFoundError:
        pass

    # update imports to the new namespace by running morpheus_namespace_update.py
    os.system(f"python {update_namespace_script} --directory {scripts_data_dir} --dfp")

    # verify the morpheus_dfp imports work
    spec, module = import_module_from_path(module_name, module_path)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        assert False, "old dfp imports are not updated to the new namespace"


@pytest.mark.usefixtures("morpheus_llm")
def test_llm_namespace_update(scripts_data_dir, update_namespace_script):
    '''
    Update the LLM namespace imports and verify the imports work.
    '''
    module_name = 'llm_old_namespace_data'
    module_path = os.path.join(scripts_data_dir, f'{module_name}.py')

    # llm imports expected to fail before namespace update
    spec, module = import_module_from_path(module_name, module_path)
    assert module is not None, f"Failed to import {module_name} from {module_path}"
    try:
        spec.loader.exec_module(module)
        assert False, "llm_namespace_data input is not setup with the old imports"
    except ModuleNotFoundError:
        pass

    # update imports to the new namespace by running morpheus_namespace_update.py
    os.system(f"python {update_namespace_script} --directory {scripts_data_dir} --llm")

    # verify the morpheus_llm imports work
    spec, module = import_module_from_path(module_name, module_path)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        assert False, "old llm imports are not updated to the new namespace"
