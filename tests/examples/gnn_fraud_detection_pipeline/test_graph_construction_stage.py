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
import typing

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from utils import TEST_DIRS


@pytest.mark.use_python
@pytest.mark.import_mod(
    [os.path.join(TEST_DIRS.examples_dir, 'gnn_fraud_detection_pipeline/stages/graph_construction_stage.py')])
class TestGraphConstructionStage:

    def test_constructor(config: Config, training_file: str, import_mod: typing.List[typing.Any]):
        graph_construction_stage = import_mod[0]
        stage = graph_construction_stage.FraudGraphConstructionStage(config, training_file)
        assert isinstance(stage._training_data, cudf.DataFrame)

        # The training datafile contains many more columns than this, but these are the four columns
        # that are depended upon in the code
        assert {'client_node', 'index', 'fraud_label', 'merchant_node'}.issubset(stage._column_names)
