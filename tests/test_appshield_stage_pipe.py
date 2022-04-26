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

import os

import pandas as pd
import pytest

from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.input.from_appshield import AppShieldSourceStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from utils import TEST_DIRS


@pytest.mark.parametrize('plugins_include', [['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']])
@pytest.mark.parametrize('cols_include', [['Base', 'Block', 'CommitCharge', 'End VPN', 'File', 'GrantedAccess',
                                            'HandleValue', 'InInit', 'InLoad', 'InMem', 'Name', 'Offset', 'PID',
                                            'Parent', 'Path', 'PrivateMemory', 'Process', 'Protection',
                                            'SHA256', 'Size', 'Start VPN', 'State', 'TID', 'Tag', 'Type',
                                            'Value', 'Variable', 'WaitReason', 'plugin', 'snapshot_id',
                                            'timestamp']])
@pytest.mark.parametrize('output_type', ['csv'])
def test_appshield_source(tmp_path, config_no_cpp, plugins_include, cols_include, output_type):

    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', '*', '*.json')
    expected_output_file = os.path.join(TEST_DIRS.tests_data_dir, 'from_appshield_results.csv')
    out_file = os.path.join(tmp_path, 'results.{}'.format(output_type))

    pipe = LinearPipeline(config_no_cpp)
    pipe.set_source(AppShieldSourceStage(config_no_cpp, input_glob, plugins_include, cols_include))
    pipe.add_stage(WriteToFileStage(config_no_cpp, filename=out_file, overwrite=False))
    pipe.run()

    assert os.path.exists(out_file)

    output_df = pd.read_csv(out_file, index_col=0)

    threadlist_df = output_df[output_df.plugin == 'threadlist']
    vadinfo_df = output_df[output_df.plugin == 'vadinfo']
    envars_df = output_df[output_df.plugin == 'envars']
    handles_df = output_df[output_df.plugin == 'handles']
    ldrmodules_df = output_df[output_df.plugin == 'ldrmodules']


    expected_output_df = pd.read_csv(expected_output_file)

    threadlist_expected_df = expected_output_df[expected_output_df.plugin == 'threadlist']
    vadinfo_expected_df = expected_output_df[expected_output_df.plugin == 'vadinfo']
    envars_expected_df = expected_output_df[expected_output_df.plugin == 'envars']
    handles_expected_df = expected_output_df[expected_output_df.plugin == 'handles']
    ldrmodules_expected_df = expected_output_df[expected_output_df.plugin == 'ldrmodules']

    assert len(threadlist_df) == len(threadlist_expected_df)
    assert len(vadinfo_df) == len(vadinfo_expected_df)
    assert len(envars_df) == len(envars_expected_df)
    assert len(handles_df) == len(handles_expected_df)
    assert len(ldrmodules_df) == len(ldrmodules_expected_df)

    inload_not_null_df = threadlist_df[~threadlist_df.InLoad.isna()]

    # length should be zero because 'InLoad' column doesn't come from threadlist
    assert len(inload_not_null_df) == 0

    inload_not_null_df = ldrmodules_df[~ldrmodules_df.InLoad.isna()]
    # length should be zero because 'InLoad' column doesn't come from threadlist
    assert len(inload_not_null_df) > 0
