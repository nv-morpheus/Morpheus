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

import os
import sys

import cupy as cp
import numpy as np
import pytest

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from utils import TEST_DIRS
from utils import assert_df_equal
from utils import get_plugin_stage_class


def build_post_proc_message(log_example_dir: str, log_test_data_dir: str):
    input_file = os.path.join(TEST_DIRS.validation_data_dir, 'log-parsing-validation-data-input.csv')
    input_df = read_file_to_df(input_file, df_type='cudf')
    meta = MessageMeta(input_df)

    # Import messages from the example dir
    import messages
    assert messages.__file__.startswith(os.path.join(log_example_dir, 'messages.py')), "Imported wrong messages module"

    # we have tensor data for the first five rows
    count = 5
    tensors = {}
    for tensor_name in ['confidences', 'input_ids', 'labels', 'seq_ids']:
        tensor_file = os.path.join(log_test_data_dir, f'{tensor_name}.csv')
        host_data = np.loadtxt(tensor_file, delimiter=',')
        tensors[tensor_name] = cp.asarray(host_data)

    memory = messages.PostprocMemoryLogParsing(count=5, **tensors)
    return messages.MultiPostprocLogParsingMessage(meta=meta,
                                                   mess_offset=0,
                                                   mess_count=count,
                                                   memory=memory,
                                                   offset=0,
                                                   count=count)


@pytest.mark.use_python
@pytest.mark.usefixtures("reset_plugins", "restore_sys_path")
def test_log_parsing_post_processing_stage(config: Config):
    config.mode = PipelineModes.NLP

    log_example_dir = os.path.join(TEST_DIRS.examples_dir, 'log_parsing')
    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'log_parsing')

    sys.path.append(log_example_dir)
    mod_path = os.path.join(log_example_dir, 'postprocessing.py')
    LogParsingPostProcessingStage = get_plugin_stage_class(mod_path, "log-postprocess", mode=config.mode)

    model_vocab_file = os.path.join(TEST_DIRS.models_dir,
                                    'training-tuning-scripts/sid-models/resources/bert-base-cased-vocab.txt')
    model_config_file = os.path.join(TEST_DIRS.tests_data_dir, 'log-parsing-config.json')

    stage = LogParsingPostProcessingStage(config, vocab_path=model_vocab_file, model_config_path=model_config_file)

    post_proc_message = build_post_proc_message(log_example_dir, log_test_data_dir)
    expected_df = read_file_to_df(os.path.join(log_test_data_dir, 'expected_out.csv'), df_type='pandas')

    print(post_proc_message)
    out_meta = stage._postprocess(post_proc_message)

    assert isinstance(out_meta, MessageMeta)
    assert_df_equal(out_meta._df, expected_df)
