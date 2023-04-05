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

import cupy as cp
import numpy as np
import pytest

import cudf

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceFILMessage
from morpheus.messages import MultiMessage
from utils import TEST_DIRS
from utils import get_plugin_stage_class


def check_inf_message(msg: MultiInferenceFILMessage,
                      expected_meta: MessageMeta,
                      expected_mess_offset: int,
                      expected_mess_count: int,
                      expected_offset: int,
                      expected_count: int,
                      expected_feature_length: int,
                      expected_flow_ids: cudf.Series,
                      expected_rollup_time: str,
                      expected_input__0: cp.ndarray):
    assert isinstance(msg, MultiInferenceFILMessage)
    assert msg.meta is expected_meta
    assert msg.mess_offset == expected_mess_offset
    assert msg.mess_count == expected_mess_count
    assert msg.offset == expected_offset
    assert msg.count == expected_count

    df = msg.get_meta()
    assert 'flow_id' in df
    assert 'rollup_time' in df

    assert (df.flow_id == expected_flow_ids).all()
    assert (df.rollup_time == expected_rollup_time).all()

    assert msg.memory.has_tensor('input__0')
    assert msg.memory.has_tensor('seq_ids')

    input__0 = msg.memory.get_tensor('input__0')
    assert input__0.shape == (expected_count, expected_feature_length)
    assert (input__0 == expected_input__0).all()

    seq_ids = msg.memory.get_tensor('seq_ids')
    assert seq_ids.shape == (expected_count, 3)
    assert (seq_ids[:, 0] == cp.arange(expected_mess_offset,
                                       expected_mess_offset + expected_mess_count,
                                       dtype=cp.uint32)).all()
    assert (seq_ids[:, 1] == 0).all()
    assert (seq_ids[:, 2] == expected_feature_length - 1).all()


@pytest.mark.usefixtures("reset_plugins")
def test_abp_pcap_preprocessing(config: Config):
    # Setup the config
    config.mode = PipelineModes.FIL
    config.feature_length = 13

    # Load the stage via the plugin manager as it isn't part of Morpheus propper
    mod_path = os.path.join(TEST_DIRS.examples_dir, 'abp_pcap_detection/abp_pcap_preprocessing.py')
    AbpPcapPreprocessingStage = get_plugin_stage_class(mod_path, "pcap-preprocess", mode=PipelineModes.FIL)

    # Get our input data, should contain the first 20 lines of the production data
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'abp_pcap.jsonlines')
    input_df = read_file_to_df(input_file, df_type='cudf', filter_nulls=False)

    expected_flow_ids = input_df.src_ip + ":" + input_df.src_port + "=" + input_df.dest_ip + ":" + input_df.dest_port
    expected_input__0 = cp.asarray(
        np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'abp_pcap_expected_input_0.csv'), delimiter=",", skiprows=0))

    assert len(input_df) == 20

    meta = MessageMeta(input_df)
    mm1 = MultiMessage(meta=meta, mess_offset=0, mess_count=10)
    mm2 = MultiMessage(meta=meta, mess_offset=10, mess_count=10)

    stage = AbpPcapPreprocessingStage(config)
    assert stage.get_needed_columns() == {'flow_id': TypeId.STRING, 'rollup_time': TypeId.STRING}

    inf1 = stage.pre_process_batch(mm1, config.feature_length, stage.features, stage.req_cols)
    check_inf_message(inf1,
                      expected_meta=meta,
                      expected_mess_offset=0,
                      expected_mess_count=10,
                      expected_offset=0,
                      expected_count=10,
                      expected_feature_length=config.feature_length,
                      expected_flow_ids=expected_flow_ids[0:10],
                      expected_rollup_time='2021-04-07 15:55',
                      expected_input__0=expected_input__0[0:10])

    inf2 = stage.pre_process_batch(mm2, config.feature_length, stage.features, stage.req_cols)
    check_inf_message(inf2,
                      expected_meta=meta,
                      expected_mess_offset=10,
                      expected_mess_count=10,
                      expected_offset=0,
                      expected_count=10,
                      expected_feature_length=config.feature_length,
                      expected_flow_ids=expected_flow_ids[10:],
                      expected_rollup_time='2021-04-07 15:55',
                      expected_input__0=expected_input__0[10:])
