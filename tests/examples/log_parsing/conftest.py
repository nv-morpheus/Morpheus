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

import importlib
import os
import sys

import pytest

from utils import TEST_DIRS


@pytest.fixture(scope="function")
@pytest.mark.usefixtures("restore_sys_path")
def log_example_dir():
    """
    Determines the location of the `log_parsing` dir, adds it to `sys.path` then yields it.
    """
    log_example_dir = os.path.join(TEST_DIRS.examples_dir, 'log_parsing')
    sys.path.append(log_example_dir)
    yield log_example_dir


def _import_mod(log_example_dir, mod_name):
    mod = importlib.import_module(mod_name)
    assert mod.__file__ == os.path.join(log_example_dir, f"{mod_name}.py")
    return mod


@pytest.fixture(scope="function")
def inference_mod(log_example_dir: str):
    yield _import_mod(log_example_dir, 'inference')


@pytest.fixture(scope="function")
def messages_mod(log_example_dir: str):
    yield _import_mod(log_example_dir, 'messages')


@pytest.fixture(scope="function")
def postprocessing_mod(log_example_dir: str):
    yield _import_mod(log_example_dir, 'postprocessing')
