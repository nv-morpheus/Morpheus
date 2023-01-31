# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import typing

import mrc

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .graph_construction_stage import FraudGraphMultiMessage


@dataclasses.dataclass
class GraphSAGEMultiMessage(MultiMessage):
    node_identifiers: typing.List[int]
    inductive_embedding_column_names: typing.List[str]


@register_stage("gnn-fraud-sage", modes=[PipelineModes.OTHER])
class GraphSAGEStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 model_hinsage_file: str,
                 batch_size: int = 5,
                 sample_size: typing.List[int] = [2, 32],
                 record_id: str = "index",
                 target_node: str = "transaction"):
        super().__init__(c)

        # Must import stellargraph before loading the model
        import stellargraph.mapper  # noqa
        import tensorflow as tf

        self._keras_model = tf.keras.models.load_model(model_hinsage_file)
        self._batch_size = batch_size
        self._sample_size = list(sample_size)
        self._record_id = record_id
        self._target_node = target_node

    @property
    def name(self) -> str:
        return "gnn-fraud-sage"

    def accepted_types(self) -> typing.Tuple:
        return (FraudGraphMultiMessage, )

    def supports_cpp_node(self):
        return False

    def _inductive_step_hinsage(
        self,
        graph,
        trained_model,
        node_identifiers,
    ):

        from stellargraph.mapper import HinSAGENodeGenerator

        # perform inductive learning from trained graph model
        # The mapper feeds data from sampled subgraph to HinSAGE model
        generator = HinSAGENodeGenerator(graph, self._batch_size, self._sample_size, head_node_type=self._target_node)
        test_gen_not_shuffled = generator.flow(node_identifiers, shuffle=False)

        inductive_emb = trained_model.predict(test_gen_not_shuffled)
        inductive_emb = cudf.DataFrame(inductive_emb, index=node_identifiers)

        return inductive_emb

    def _process_message(self, message: FraudGraphMultiMessage):
        node_identifiers = list(message.get_meta(self._record_id).to_pandas())

        inductive_embedding = self._inductive_step_hinsage(message.graph, self._keras_model, node_identifiers)

        # Rename the columns to be more descriptive
        inductive_embedding.rename(lambda x: "ind_emb_" + str(x), axis=1, inplace=True)

        for col in inductive_embedding.columns.values.tolist():
            message.set_meta(col, inductive_embedding[col])

        assert (message.mess_count == len(inductive_embedding))

        return GraphSAGEMultiMessage(meta=message.meta,
                                     node_identifiers=node_identifiers,
                                     inductive_embedding_column_names=inductive_embedding.columns.values.tolist(),
                                     mess_offset=message.mess_offset,
                                     mess_count=message.mess_count)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self._process_message)
        builder.make_edge(input_stream[0], node)
        return node, GraphSAGEMultiMessage
