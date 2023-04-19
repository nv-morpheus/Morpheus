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

import typing

import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.utils import compare_df
from utils import TEST_DIRS
from utils import assert_results
from utils.dataset_manager import DatasetManager


@pytest.mark.use_python
class TestGraphSageStage:

    def test_constructor(config: Config, hinsage_model: str, gnn_fraud_detection_pipeline: typing.Any, tensorflow):
        from gnn_fraud_detection_pipeline.stages.graph_sage_stage import GraphSAGEStage
        stage = GraphSAGEStage(config,
                               model_hinsage_file=hinsage_model,
                               batch_size=10,
                               sample_size=[4, 64],
                               record_id="test_id",
                               target_node="test_node")

        assert isinstance(stage._keras_model, tensorflow.keras.models.Model)
        assert stage._batch_size == 10
        assert stage._sample_size == [4, 64]
        assert stage._record_id == "test_id"
        assert stage._target_node == "test_node"

    def test_inductive_step_hinsage(config: Config,
                                    hinsage_model: str,
                                    gnn_fraud_detection_pipeline: typing.Any,
                                    test_data: dict,
                                    dataset_pandas: DatasetManager):
        from gnn_fraud_detection_pipeline.stages.graph_construction_stage import FraudGraphConstructionStage
        from gnn_fraud_detection_pipeline.stages.graph_sage_stage import GraphSAGEStage

        # The column names in the saved test data will be strings, in the results they will be ints
        expected_df = dataset_pandas['examples/gnn_fraud_detection_pipeline/inductive_emb.csv']
        expected_df.rename(lambda x: int(x), axis=1, inplace=True)

        df = test_data['df']

        graph = FraudGraphConstructionStage._build_graph_features(df)

        stage = GraphSAGEStage(config, model_hinsage_file=hinsage_model)
        results = stage._inductive_step_hinsage(graph, stage._keras_model, test_data['index'])

        assert isinstance(results, cudf.DataFrame)
        assert results.index.to_arrow().to_pylist() == test_data['index']
        assert_results(compare_df.compare_df(results.to_pandas(), expected_df))

    def test_process_message(config: Config,
                             hinsage_model: str,
                             gnn_fraud_detection_pipeline: typing.Any,
                             test_data: dict,
                             dataset_pandas: DatasetManager):
        from gnn_fraud_detection_pipeline.stages.graph_construction_stage import FraudGraphConstructionStage
        from gnn_fraud_detection_pipeline.stages.graph_construction_stage import FraudGraphMultiMessage
        from gnn_fraud_detection_pipeline.stages.graph_sage_stage import GraphSAGEMultiMessage
        from gnn_fraud_detection_pipeline.stages.graph_sage_stage import GraphSAGEStage

        expected_df = dataset_pandas['examples/gnn_fraud_detection_pipeline/inductive_emb.csv']
        expected_df.rename(lambda x: "ind_emb_{}".format(x), axis=1, inplace=True)

        df = test_data['df']
        meta = MessageMeta(cudf.DataFrame(df))
        graph = FraudGraphConstructionStage._build_graph_features(df)
        msg = FraudGraphMultiMessage(meta=meta, graph=graph)

        stage = GraphSAGEStage(config, model_hinsage_file=hinsage_model)
        results = stage._process_message(msg)

        assert isinstance(results, GraphSAGEMultiMessage)
        assert results.meta is meta
        assert results.mess_offset == 0
        assert results.mess_count == len(df)
        assert results.node_identifiers == test_data['index']
        assert sorted(results.inductive_embedding_column_names) == sorted(expected_df.columns)

        ind_emb_df = results.get_meta(results.inductive_embedding_column_names)
        assert_results(compare_df.compare_df(ind_emb_df.to_pandas(), expected_df))
