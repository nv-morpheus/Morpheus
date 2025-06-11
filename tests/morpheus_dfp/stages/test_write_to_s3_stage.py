# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    from morpheus_dfp.stages.write_to_s3_stage import WriteToS3Stage

    mock_s3_writer = mock.MagicMock()
    stage = WriteToS3Stage(config, s3_writer=mock_s3_writer)

    assert isinstance(stage, SinglePortStage)
    assert stage._s3_writer is mock_s3_writer
