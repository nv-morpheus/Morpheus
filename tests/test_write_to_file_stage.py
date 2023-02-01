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

import os
from unittest import mock

import pytest

from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from utils import TEST_DIRS
from utils import assert_path_exists


@pytest.mark.use_python
@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("output_type", ["csv", "json", "jsonlines"])
def test_file_rw_pipe(tmp_path, config, output_type, flush):
    """
    Test the flush functionality of the WriteToFileStage.
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    # This currently works because the FileSourceStage doesn't use the builtin open function, but WriteToFileStage does
    mock_open = mock.mock_open()
    with mock.patch('builtins.open', mock_open):
        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=input_file))
        pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False, flush=flush))
        pipe.run()

    assert not os.path.exists(out_file)
    assert mock_open().flush.called == flush
