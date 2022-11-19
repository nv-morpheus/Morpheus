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

import json
import os
from unittest import mock

import morpheus
import morpheus.config

# Using morpheus.config to distinguish between the config package and the pytest fixture of the same name


@mock.patch('docker.from_env')
def test_auto_determine_bootstrap(mock_docker_from_env):
    mock_net = mock.MagicMock()
    mock_net.get.return_value = mock_net
    mock_net.attrs = {'IPAM': {'Config': [{'Gateway': 'test_bridge_ip'}]}}

    # create mock containers
    mc1 = mock.MagicMock()
    mc1.ports = {'9092/tcp': [{'HostIp': 'test_host1', 'HostPort': '47'}]}

    mc2 = mock.MagicMock()
    mc2.ports = {}

    mc3 = mock.MagicMock()
    mc3.ports = {'9092/tcp': [{'HostIp': 'test_host2', 'HostPort': '42'}]}

    mock_net.containers = [mc1, mc2, mc3]

    mock_docker_client = mock.MagicMock()
    mock_docker_client.networks = mock_net
    mock_docker_from_env.return_value = mock_docker_client

    bootstrap_servers = morpheus.config.auto_determine_bootstrap()
    assert bootstrap_servers == "test_bridge_ip:47,test_bridge_ip:42"


def test_config_base(config):
    assert isinstance(config, morpheus.config.ConfigBase)


def test_config_onnx_to_trt():
    c = morpheus.config.ConfigOnnxToTRT(input_model='frogs',
                                        output_model='toads',
                                        batches=[(1, 2), (3, 4)],
                                        seq_length=100,
                                        max_workspace_size=512)

    assert isinstance(c, morpheus.config.ConfigBase)
    assert isinstance(c, morpheus.config.ConfigOnnxToTRT)
    assert c.input_model == 'frogs'
    assert c.output_model == 'toads'
    assert c.batches == [(1, 2), (3, 4)]
    assert c.seq_length == 100
    assert c.max_workspace_size == 512


def test_auto_encoder():
    c = morpheus.config.ConfigAutoEncoder(feature_columns=['a', 'b', 'c', 'def'],
                                          userid_column_name='def',
                                          userid_filter='testuser')

    assert isinstance(c, morpheus.config.ConfigBase)
    assert isinstance(c, morpheus.config.ConfigAutoEncoder)
    assert c.feature_columns == ['a', 'b', 'c', 'def']
    assert c.userid_column_name == 'def'
    assert c.userid_filter == 'testuser'


def test_pipeline_modes():
    expected = {"OTHER", "NLP", "FIL", "AE"}
    entries = set(pm.name for pm in morpheus.config.PipelineModes)
    assert entries.issuperset(expected)


def test_config_save(tmp_path):
    filename = os.path.join(tmp_path, 'config.json')

    c = morpheus.config.Config()
    c.save(filename)

    assert os.path.exists(filename)
    with open(filename) as fh:
        assert isinstance(json.load(fh), dict)


def test_to_string(config):
    s = config.to_string()
    assert isinstance(s, str)
    assert isinstance(json.loads(s), dict)
