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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithe r express or implied.
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
class TestClassificationStage:

    def test_constructor(config: Config, xgb_model: str, gnn_fraud_detection_pipeline: typing.Any, cuml: typing.Any):
        from gnn_fraud_detection_pipeline.stages.classification_stage import ClassificationStage

        stage = ClassificationStage(config, xgb_model)
        assert isinstance(stage._xgb_model, cuml.ForestInference)

    def test_process_message(config: Config, xgb_model: str, gnn_fraud_detection_pipeline: typing.Any, dataset_cudf):
        from gnn_fraud_detection_pipeline.stages.classification_stage import ClassificationStage
        from gnn_fraud_detection_pipeline.stages.graph_sage_stage import GraphSAGEMultiMessage

        df = dataset_cudf['examples/gnn_fraud_detection_pipeline/inductive_emb.csv']
        df.rename(lambda x: "ind_emb_{}".format(x), axis=1, inplace=True)

        expected_df = dataset_cudf.pandas['examples/gnn_fraud_detection_pipeline/predictions.csv']
        assert len(df) == len(expected_df)

        # The exact values of the node_identifiers aren't important to this stage, we just need to verify that they're
        # inserted into a "node_id" column in the DF
        node_identifiers = expected_df['node_id'].tolist()

        ind_emb_columns = list(df.columns)

        meta = MessageMeta(df)
        msg = GraphSAGEMultiMessage(meta=meta,
                                    node_identifiers=node_identifiers,
                                    inductive_embedding_column_names=ind_emb_columns)

        stage = ClassificationStage(config, xgb_model)
        results = stage._process_message(msg)
        print(results.get_meta(['prediction', 'node_id']))

        # The stage actually edits the message in place, and returns it, but we don't need to assert that
        assert_results(compare_df.compare_df(results.get_meta(['prediction', 'node_id']).to_pandas(), expected_df))
