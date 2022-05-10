import typing

import neo

import cuml

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .graph_sage_stage import GraphSAGEMultiMessage


class ClassificationStage(SinglePortStage):

    def __init__(self, c: Config, model_xgb_file: str):
        super().__init__(c)

        self._xgb_model = cuml.ForestInference.load(model_xgb_file, output_class=True)

    @property
    def name(self) -> str:
        return "gnn-fraud-classification"

    def accepted_types(self) -> typing.Tuple:
        return (GraphSAGEMultiMessage, )

    def _process_message(self, message: GraphSAGEMultiMessage):
        ind_emb_columns = message.get_meta(message.inductive_embedding_column_names)

        message.set_meta("node_id", message.node_identifiers)

        prediction = self._xgb_model.predict_proba(ind_emb_columns).iloc[:, 1]

        message.set_meta("prediction", prediction)

        return message

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        node = seg.make_node(self.unique_name, self._process_message)
        seg.make_edge(input_stream[0], node)
        return node, MultiMessage
