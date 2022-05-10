import dataclasses
import typing

import neo
import tensorflow as tf
from stellargraph.mapper import HinSAGENodeGenerator

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .graph_construction_stage import FraudGraphMultiMessage


@dataclasses.dataclass
class GraphSAGEMultiMessage(MultiMessage):
    node_identifiers: typing.List[int]
    inductive_embedding_column_names: typing.List[str]


class GraphSAGEStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 model_hinsage_file: str,
                 batch_size: int = 5,
                 sample_size=[2, 32],
                 record_id: str = "index",
                 target_node: str = "transaction"):
        super().__init__(c)
        self._keras_model = tf.keras.models.load_model(model_hinsage_file)
        self._batch_size = batch_size
        self._sample_size = sample_size
        self._record_id = record_id
        self._target_node = target_node

    @property
    def name(self) -> str:
        return "gnn-fraud-sage"

    def accepted_types(self) -> typing.Tuple:
        return (FraudGraphMultiMessage, )

    def _inductive_step_hinsage(
        self,
        graph,
        trained_model,
        node_identifiers,
    ):
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

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        node = seg.make_node(self.unique_name, self._process_message)
        seg.make_edge(input_stream[0], node)
        return node, GraphSAGEMultiMessage
