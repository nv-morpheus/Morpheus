import threading
import typing

import mrc

from morpheus.config import Config
from morpheus.pipeline.pass_thru_type_mixin import InferredPassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


class RecordThreadIdStage(InferredPassThruTypeMixin, SinglePortStage):
    """
    Forwarding stage that records the thread id of the progress engine
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.thread_id = None

    @property
    def name(self):
        return "record-thread"

    def accepted_types(self):
        return (typing.Any, )

    def supports_cpp_node(self):
        return False

    def _save_thread(self, x):
        self.thread_id = threading.current_thread().ident
        return x

    def _build_single(self, builder: mrc.Builder, input_stream):
        stream = builder.make_node(self.unique_name, mrc.core.operators.map(self._save_thread))

        builder.make_edge(input_stream[0], stream)

        return stream, input_stream[1]