# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import types

import pytest

from _utils import TEST_DIRS


@pytest.fixture(scope="function")
def import_vdb_update_utils_module(restore_sys_path, pymilvus: types.ModuleType):  # pylint: disable=unused-argument
    path = os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/')
    sys.path.insert(0, path)

    import vdb_utils

    return vdb_utils


@pytest.fixture(scope="function")
def import_schema_transform_module(restore_sys_path):  # pylint: disable=unused-argument
    path = os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module')
    sys.path.insert(0, path)

    import schema_transform

    return schema_transform
