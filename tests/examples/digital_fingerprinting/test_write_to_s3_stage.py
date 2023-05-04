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

from unittest import mock

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage


def test_constructor(config: Config):
    from dfp.stages.write_to_s3_stage import WriteToS3Stage

    mock_s3_writer = mock.MagicMock()
    stage = WriteToS3Stage(config, s3_writer=mock_s3_writer)

    assert isinstance(stage, SinglePortStage)
    assert stage._s3_writer is mock_s3_writer


@mock.patch("dfp.stages.write_to_s3_stage.ops")
def test_build_single(mock_mrc_ops, config: Config):
    from dfp.stages.write_to_s3_stage import WriteToS3Stage

    mock_s3_writer = mock.MagicMock()
    stage = WriteToS3Stage(config, s3_writer=mock_s3_writer)

    mock_mrc_map_fn = mock.MagicMock()
    mock_mrc_ops.map.return_value = mock_mrc_map_fn

    input_stream = (mock.MagicMock(), mock.MagicMock())

    mock_node = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_builder.make_node.return_value = mock_node

    results = stage._build_single(mock_builder, input_stream)

    assert results == (mock_node, input_stream[1])
    mock_builder.make_node.assert_called_once_with(stage.unique_name, mock_mrc_map_fn)
    mock_mrc_ops.map.assert_called_once_with(mock_s3_writer)
    mock_builder.make_edge.assert_called_once_with(input_stream[0], mock_node)
