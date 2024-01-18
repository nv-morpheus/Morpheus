# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from _utils import TEST_DIRS


@pytest.fixture(scope="module")
def import_vdb_update_utils_module():
    path = os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/')
    sys.path.insert(0, path)

    import vdb_utils
    sys.path.remove(path)

    return vdb_utils


@pytest.fixture(scope="module")
def import_schema_transform_module():
    path = os.path.join(TEST_DIRS.examples_dir, 'llm/vdb_upload/module')
    sys.path.insert(0, path)

    import schema_transform
    sys.path.remove(path)

    return schema_transform
