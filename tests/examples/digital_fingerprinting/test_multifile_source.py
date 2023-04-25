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
import os
import types
import typing
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.pipeline.single_output_source import SingleOutputSource
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


def test_constructor(config: Config):
    from dfp.stages.multi_file_source import MultiFileSource

    batch_size = 1234
    n_threads = 13
    config.pipeline_batch_size = batch_size
    config.num_threads = n_threads
    filenames = ['some/file', '/tmp/some/files-2023-*-*.csv', 's3://some/bucket/2023-*-*.csv.gz']
    stage = MultiFileSource(config, filenames=filenames)

    assert isinstance(stage, SingleOutputSource)
    assert stage._batch_size == batch_size
    assert stage._max_concurrent == n_threads
    assert stage._filenames == filenames
