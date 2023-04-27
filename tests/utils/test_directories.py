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

import os

import morpheus


class TestDirectories(object):

    def __init__(self, cur_file=__file__) -> None:
        self.tests_dir = os.path.dirname(os.path.dirname(cur_file))
        self.morpheus_root = os.environ.get('MORPHEUS_ROOT', os.path.dirname(self.tests_dir))
        self.data_dir = morpheus.DATA_DIR
        self.examples_dir = os.path.join(self.morpheus_root, 'examples')
        self.models_dir = os.path.join(self.morpheus_root, 'models')
        self.datasets_dir = os.path.join(self.models_dir, 'datasets')
        self.training_data_dir = os.path.join(self.datasets_dir, 'training-data')
        self.validation_data_dir = os.path.join(self.datasets_dir, 'validation-data')
        self.tests_data_dir = os.path.join(self.tests_dir, 'tests_data')
        self.mock_triton_servers_dir = os.path.join(self.tests_dir, 'mock_triton_server')
