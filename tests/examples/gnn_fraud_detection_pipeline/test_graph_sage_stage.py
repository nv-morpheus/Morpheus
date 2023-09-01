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

import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage

# pylint: disable=no-name-in-module


@pytest.mark.use_python
class TestGraphSageStage:

    def test_constructor(self, config: Config, model_dir: str):
        from stages.graph_sage_stage import GraphSAGEStage
        from stages.model import HinSAGE
        stage = GraphSAGEStage(config, model_dir=model_dir, batch_size=10, record_id="test_id", target_node="test_node")

        assert isinstance(stage._dgl_model, HinSAGE)
        assert stage._batch_size == 10
        assert stage._record_id == "test_id"
        assert stage._target_node == "test_node"

    def test_process_message(self,
                             config: Config,
                             training_file: str,
                             model_dir: str,
                             test_data: dict,
                             dataset_pandas: DatasetManager):
        from stages.graph_construction_stage import FraudGraphConstructionStage
        from stages.graph_sage_stage import GraphSAGEMultiMessage
        from stages.graph_sage_stage import GraphSAGEStage

        expected_df = dataset_pandas['examples/gnn_fraud_detection_pipeline/inductive_emb.csv']

        df = test_data['df']
        meta = MessageMeta(cudf.DataFrame(df))
        multi_msg = MultiMessage(meta=meta)
        construction_stage = FraudGraphConstructionStage(config, training_file)
        fgmm_msg = construction_stage._process_message(multi_msg)

        stage = GraphSAGEStage(config, model_dir=model_dir)
        results = stage._process_message(fgmm_msg)

        assert isinstance(results, GraphSAGEMultiMessage)
        assert results.meta is meta
        assert results.mess_offset == 0
        assert results.mess_count == len(df)
        assert results.node_identifiers == test_data['index']

        cols = results.inductive_embedding_column_names + ['index']
        assert sorted(cols) == sorted(expected_df.columns)
        ind_emb_df = results.get_meta(cols)
        dataset_pandas.assert_compare_df(ind_emb_df.to_pandas(), expected_df)
