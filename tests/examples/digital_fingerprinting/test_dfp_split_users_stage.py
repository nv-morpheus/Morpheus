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

import json
import os
import typing

import pytest

from _utils import TEST_DIRS
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage


def test_constructor(config: Config):
    from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage

    stage = DFPSplitUsersStage(config, include_generic=False, include_individual=True)
    assert isinstance(stage, SinglePortStage)
    assert not stage._include_generic
    assert stage._include_individual
    assert not stage._skip_users
    assert not stage._only_users
    assert not stage._user_index_map

    stage = DFPSplitUsersStage(config,
                               include_generic=True,
                               include_individual=False,
                               skip_users=['a', 'b'],
                               only_users=['c', 'd'])

    assert stage._include_generic
    assert not stage._include_individual
    assert stage._skip_users == ['a', 'b']
    assert stage._only_users == ['c', 'd']
    assert not stage._user_index_map


@pytest.mark.parametrize('include_generic', [True, False])
@pytest.mark.parametrize('include_individual', [True, False])
@pytest.mark.parametrize(
    'skip_users',
    [[], ['katherine-malton@forward.org'], ['terrietahon@planner-viral.com', 'angelaethridge@lomo-customer.net']])
@pytest.mark.parametrize('only_users',
                         [[], ['WENDY.HUERTA@Brooklyn-dynamic.edu'],
                          ['terrietahon@planner-viral.com', 'SAMUEL.DAVIS@transition-high-life.com']])
def test_extract_users(config: Config,
                       dataset: DatasetManager,
                       include_generic: bool,
                       include_individual: bool,
                       skip_users: typing.List[str],
                       only_users: typing.List[str]):
    from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
    config.ae.userid_column_name = "From"
    config.ae.fallback_username = "testy_testerson"

    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              "examples/developer_guide/email_with_addresses_first_10.jsonlines")

    df = dataset[input_file]

    all_data = []
    expected_data = {}
    with open(input_file, encoding='UTF-8') as fh:
        for line in fh:
            json_data = json.loads(line)
            user_id = json_data['From']
            if user_id in skip_users:
                continue

            if len(only_users) > 0 and user_id not in only_users:
                continue

            if include_generic:
                all_data.append(json_data)

            if include_individual:
                expected_data[user_id] = [json_data]

    if include_generic:
        expected_data[config.ae.fallback_username] = all_data

    stage = DFPSplitUsersStage(config,
                               include_generic=include_generic,
                               include_individual=include_individual,
                               skip_users=skip_users,
                               only_users=only_users)

    results = stage.extract_users(df)

    if not include_generic and not include_individual:
        # Extra check for weird combination
        assert len(results) == 0

    # Add one for the generic user
    assert len(results) == len(expected_data)
    for msg in results:
        assert len(msg.df) == len(expected_data[msg.user_id])
        if msg.user_id != config.ae.fallback_username:
            assert msg.df.iloc[0].to_dict() == expected_data[msg.user_id][0]


def test_extract_users_none_to_empty(config: Config):
    from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage

    stage = DFPSplitUsersStage(config, include_generic=True, include_individual=True)
    assert not stage.extract_users(None)
