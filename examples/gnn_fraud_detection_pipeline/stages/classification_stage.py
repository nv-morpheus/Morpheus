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

import typing

import srf

import cuml

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .graph_sage_stage import GraphSAGEMultiMessage


@register_stage("gnn-fraud-classification", modes=[PipelineModes.OTHER])
class ClassificationStage(SinglePortStage):

    def __init__(self, c: Config, model_xgb_file: str):
        """
        The GNN Fraud Classification Stage

        Parameters
        ----------
        c : Config
            The Morpheus global config object
        model_xgb_file : str
            The XGB model to load
        """

        super().__init__(c)

        self._xgb_model = cuml.ForestInference.load(model_xgb_file, output_class=True)

    @property
    def name(self) -> str:
        return "gnn-fraud-classification"

    def accepted_types(self) -> typing.Tuple:
        return (GraphSAGEMultiMessage, )

    def supports_cpp_node(self):
        return False

    def _process_message(self, message: GraphSAGEMultiMessage):
        ind_emb_columns = message.get_meta(message.inductive_embedding_column_names)

        message.set_meta("node_id", message.node_identifiers)

        prediction = self._xgb_model.predict_proba(ind_emb_columns).iloc[:, 1]

        message.set_meta("prediction", prediction)

        return message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self._process_message)
        builder.make_edge(input_stream[0], node)
        return node, MultiMessage
