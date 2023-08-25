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

import logging
from unittest import mock

import numpy as np
import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.logger import set_log_level


def test_constructor(config: Config):
    from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
    stage = DFPPostprocessingStage(config)
    assert isinstance(stage, SinglePortStage)
    assert stage._needed_columns['event_time'] == TypeId.STRING


@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.parametrize('use_on_data', [True, False])
@pytest.mark.parametrize('morpheus_log_level',
                         [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG])
@mock.patch('dfp.stages.dfp_postprocessing_stage.datetime')
def test_process_events_on_data(mock_datetime: mock.MagicMock,
                                config: Config,
                                dfp_multi_ae_message: MultiAEMessage,
                                use_on_data: bool,
                                morpheus_log_level: int):
    from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage

    mock_dt_obj = mock.MagicMock()
    mock_dt_obj.strftime.return_value = '2021-01-01T00:00:00Z'
    mock_datetime.now.return_value = mock_dt_obj

    # post-process should replace nans, lets add a nan to the DF
    with dfp_multi_ae_message.meta.mutable_dataframe() as df:
        df['v2'][10] = np.nan
        df['event_time'] = ''

    set_log_level(morpheus_log_level)
    stage = DFPPostprocessingStage(config)

    # on_data is a thin wrapper around process_events, tests should be the same for non-empty messages
    if use_on_data:
        assert stage.on_data(dfp_multi_ae_message) is dfp_multi_ae_message
    else:
        stage._process_events(dfp_multi_ae_message)

    assert isinstance(dfp_multi_ae_message, MultiAEMessage)
    result_df = dfp_multi_ae_message.meta.copy_dataframe()
    assert (result_df['event_time'] == '2021-01-01T00:00:00Z').all()
    assert result_df['v2'][10] == 'NaN'


def test_on_data_none(config: Config):
    from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
    stage = DFPPostprocessingStage(config)
    assert stage.on_data(None) is None
    assert stage.on_data(mock.MagicMock(mess_count=0)) is None
